"""VadCLIP residual adapter with an exact-zero, trainable residual scale.

The supplied VadCLIP baseline is held frozen.  Unlike the earlier gated
adapter, this module keeps the residual MLP non-zero at construction and
initialises only its scalar scale to zero.  The first forward pass is therefore
exactly the baseline, while the scale still receives a useful gradient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def add_vadclip_source(vadclip_root: str) -> None:
    """Make the unchanged official VadCLIP source importable."""
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def split_concat(visual: torch.Tensor, neuron_width: int, clip_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and split the repository's [selected neuron | CLIP] contract."""
    expected = int(neuron_width) + int(clip_dim)
    if visual.ndim != 3 or visual.shape[-1] != expected:
        raise ValueError(f"expected [B,T,{expected}] [neuron|clip], got {tuple(visual.shape)}")
    return visual[..., :neuron_width], visual[..., neuron_width:]


class ZeroStartNormalAnchorVadCLIP(nn.Module):
    """Inject a trainable neuron residual before an otherwise frozen VadCLIP.

    ``residual_scale`` starts at zero, so initial predictions match the supplied
    baseline exactly.  The MLP retains PyTorch's non-zero default initialisation;
    consequently ``residual_scale`` has a non-vanishing gradient at step one.
    """

    def __init__(
        self,
        options,
        vadclip_root: str,
        device: str,
        neuron_width: int = 768,
        clip_dim: int = 512,
        residual_hidden_dim: int = 1024,
        residual_depth: int = 3,
        residual_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        if clip_dim != 512:
            raise ValueError("VadCLIP's final CLIP interface is fixed at 512D")
        if residual_depth < 1 or residual_hidden_dim <= 0:
            raise ValueError("residual-depth and residual-hidden-dim must be positive")
        add_vadclip_source(vadclip_root)
        from model import CLIPVAD

        self.neuron_width, self.clip_dim = int(neuron_width), int(clip_dim)
        self.neuron_norm = nn.LayerNorm(self.neuron_width)
        dimensions = [self.neuron_width] + [int(residual_hidden_dim)] * (int(residual_depth) - 1) + [self.clip_dim]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.append(nn.GELU())
        self.neuron_to_clip = nn.Sequential(*layers)
        # Do not zero the final layer: with a zero residual scale this remains
        # baseline-equivalent, but the scale can learn immediately.
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        self.base = CLIPVAD(
            options.classes_num, options.embed_dim, options.visual_length, self.clip_dim,
            options.visual_head, options.visual_layers, options.attn_window,
            options.prompt_prefix, options.prompt_postfix, device,
        )
        self._base_frozen = False

    def freeze_base(self) -> None:
        """Freeze only the official VadCLIP parameters."""
        self.base.requires_grad_(False)
        self._base_frozen = True

    def train(self, mode: bool = True):
        """Retain the official backbone's train/eval transitions."""
        super().train(mode)
        return self

    def residual_scale_value(self) -> torch.Tensor:
        """Return the signed, directly optimised injection scale."""
        return self.residual_scale

    def forward_with_residual(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        """Return official outputs plus the injected residual for normal anchoring."""
        neuron, clip = split_concat(visual, self.neuron_width, self.clip_dim)
        correction = self.neuron_to_clip(self.neuron_norm(neuron))
        residual = self.residual_scale.to(clip.dtype) * correction.to(clip.dtype)
        text_features, logits1, logits2 = self.base(clip + residual, padding_mask, text, lengths)
        return text_features, logits1, logits2, residual

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        """Keep the ordinary VadCLIP three-output inference interface."""
        text_features, logits1, logits2, _residual = self.forward_with_residual(visual, padding_mask, text, lengths)
        return text_features, logits1, logits2


def build_residual_model(
    options,
    vadclip_root: str,
    device: str,
    contract: dict,
    residual_hidden_dim: int,
    residual_depth: int,
    residual_scale_init: float = 0.0,
) -> ZeroStartNormalAnchorVadCLIP:
    """Build an adapter only when the selected-neuron concat contract is valid."""
    neuron_width = int(contract.get("neuron_width", 768))
    clip_dim = int(contract.get("clip_dim", 512))
    input_width = int(contract.get("input_width", contract.get("visual_width", neuron_width + clip_dim)))
    if input_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    return ZeroStartNormalAnchorVadCLIP(
        options,
        vadclip_root,
        device,
        neuron_width,
        clip_dim,
        residual_hidden_dim,
        residual_depth,
        residual_scale_init,
    )
