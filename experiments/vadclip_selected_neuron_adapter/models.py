"""Frozen VadCLIP plus small trainable adapters at selected CLIP neurons."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from common import add_vadclip_source, load_selected_dims, state_dict_from_file


class SelectedNeuronAdapter(nn.Module):
    """A zero-start bottleneck which changes only selected CLS coordinates.

    The base CLIP block is untouched and frozen.  The adapter receives the
    post-block CLS token, predicts a delta for just the selected coordinates,
    and feeds that modified CLS token into the next frozen CLIP block.
    """

    def __init__(self, dimensions: Iterable[int], rank: int) -> None:
        super().__init__()
        selected = torch.as_tensor(list(dimensions), dtype=torch.long)
        if selected.ndim != 1 or selected.numel() == 0:
            raise ValueError("an adapter needs at least one selected neuron")
        bottleneck = min(int(rank), int(selected.numel()))
        if bottleneck <= 0:
            raise ValueError("adapter rank must be positive")
        self.register_buffer("dimensions", selected, persistent=True)
        self.norm = nn.LayerNorm(selected.numel())
        self.down = nn.Linear(selected.numel(), bottleneck)
        self.up = nn.Linear(bottleneck, selected.numel())
        # This makes the complete visual encoder exactly the original CLIP at
        # step zero, while still allowing gradients into ``up`` on step one.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != 768:
            raise ValueError(f"expected [tokens,batch,768], got {tuple(tokens.shape)}")
        cls = tokens[0]
        selected = cls.index_select(1, self.dimensions)
        delta = self.up(torch.nn.functional.gelu(self.down(self.norm(selected.float()))))
        update = torch.zeros_like(cls).index_copy(1, self.dimensions, delta.to(cls.dtype))
        # Do not modify ``tokens`` in place: it is used by autograd inside the
        # following frozen transformer blocks.
        return torch.cat((cls.add(update).unsqueeze(0), tokens[1:]), dim=0)


class SelectedNeuronAdapterVadCLIP(nn.Module):
    """Run image frames through frozen CLIP and then the frozen VadCLIP head."""

    def __init__(self, options, vadclip_root: str, device: str, neuron_json: str, adapter_rank: int) -> None:
        super().__init__()
        add_vadclip_source(vadclip_root)
        from model import CLIPVAD

        self.base = CLIPVAD(
            options.classes_num, options.embed_dim, options.visual_length, 512,
            options.visual_head, options.visual_layers, options.attn_window,
            options.prompt_prefix, options.prompt_postfix, device,
        )
        dimensions = load_selected_dims(neuron_json)
        visual = self.base.clipmodel.visual
        actual_layers = len(visual.transformer.resblocks)
        if actual_layers != len(dimensions):
            raise ValueError(f"selection has {len(dimensions)} layers but CLIP has {actual_layers}")
        self.adapters = nn.ModuleList(SelectedNeuronAdapter(dims, adapter_rank) for dims in dimensions)
        self._base_frozen = False

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self._base_frozen = True

    def trainable_adapter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Manual ViT-B/16 forward with adapters after the selected blocks."""
        visual = self.base.clipmodel.visual
        if images.ndim != 4:
            raise ValueError(f"expected [frames,3,H,W], got {tuple(images.shape)}")
        x = images.to(dtype=visual.conv1.weight.dtype)
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat((cls, x), dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x).permute(1, 0, 2)
        for block, adapter in zip(visual.transformer.resblocks, self.adapters):
            x = block(x)
            x = adapter(x)
        x = visual.ln_post(x.permute(1, 0, 2)[:, 0, :])
        if visual.proj is not None:
            x = x @ visual.proj
        return x

    def encode_frame_sequence(self, frames: torch.Tensor, frame_batch_size: int) -> torch.Tensor:
        """Encode all snippets of one video without changing their order."""
        if frame_batch_size <= 0:
            raise ValueError("frame_batch_size must be positive")
        pieces = []
        for left in range(0, frames.shape[0], frame_batch_size):
            pieces.append(self.encode_images(frames[left:left + frame_batch_size]))
        return torch.cat(pieces, dim=0)

    def encode_feature_anchored_sequence(
        self,
        frames: torch.Tensor,
        source_feature: torch.Tensor,
        frame_batch_size: int,
    ) -> torch.Tensor:
        """Apply the online Adapter difference to the original 512D cache.

        The released XD 512D archive cannot be reproduced exactly from the
        available raw videos, although the selected hidden cache is aligned.
        Anchoring preserves the original VadCLIP input at zero Adapter while
        retaining a differentiable change produced inside frozen CLIP blocks.
        """
        if source_feature.ndim != 2 or source_feature.shape[1] != 512:
            raise ValueError(f"expected source [T,512] feature, got {tuple(source_feature.shape)}")
        adapted_pieces, frozen_pieces = [], []
        for left in range(0, frames.shape[0], frame_batch_size):
            image_batch = frames[left:left + frame_batch_size]
            adapted_pieces.append(self.encode_images(image_batch))
            with torch.no_grad():
                frozen_pieces.append(self.base.clipmodel.encode_image(image_batch))
        adapted = torch.cat(adapted_pieces, dim=0)
        frozen = torch.cat(frozen_pieces, dim=0).to(adapted.dtype)
        if source_feature.shape != adapted.shape:
            raise ValueError(
                f"source cache shape {tuple(source_feature.shape)} and online CLIP shape {tuple(adapted.shape)} differ"
            )
        anchor = source_feature.to(device=adapted.device, dtype=adapted.dtype)
        return anchor + (adapted - frozen)

    def forward(self, visual_features: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        return self.base(visual_features, padding_mask, text, lengths)


def initialize_frozen_baseline(model: SelectedNeuronAdapterVadCLIP, baseline_path: str) -> int:
    """Copy an official VadCLIP checkpoint into the wrapper's ``base`` only."""
    source, target = state_dict_from_file(baseline_path), model.state_dict()
    missing, mismatched, copied = [], [], 0
    for name, tensor in target.items():
        if not name.startswith("base."):
            continue
        candidate_name = name.removeprefix("base.")
        candidate = source.get(candidate_name)
        if candidate is None:
            missing.append(candidate_name)
        elif tuple(candidate.shape) != tuple(tensor.shape):
            mismatched.append((candidate_name, tuple(candidate.shape), tuple(tensor.shape)))
        else:
            target[name] = candidate
            copied += 1
    if missing or mismatched:
        raise RuntimeError(
            "baseline checkpoint is incompatible with official 512D VadCLIP: "
            f"missing={missing[:4]}, mismatched={mismatched[:2]}"
        )
    model.load_state_dict(target, strict=True)
    model.freeze_base()
    return copied


def build_model(options, vadclip_root: str, device: str, neuron_json: str, adapter_rank: int) -> SelectedNeuronAdapterVadCLIP:
    return SelectedNeuronAdapterVadCLIP(options, vadclip_root, device, neuron_json, adapter_rank)
