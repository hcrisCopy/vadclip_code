"""Direct global-768 score residual model; official VadCLIP source is unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as functional


def add_vadclip_source(vadclip_root: str) -> None:
    """Make the unmodified official VadCLIP source importable."""
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def split_concat(visual: torch.Tensor, neuron_width: int, clip_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and split the repository's [neuron | CLIP] feature contract."""
    expected = int(neuron_width) + int(clip_dim)
    if visual.ndim != 3 or visual.shape[-1] != expected:
        raise ValueError(f"expected [B,T,{expected}] [neuron|clip], got {tuple(visual.shape)}")
    return visual[..., :neuron_width], visual[..., neuron_width:]


class TemporalScoreAdapter(nn.Module):
    """Predict bounded per-snippet logit corrections from selected neurons.

    The two output layers start at zero.  Therefore the first forward pass is
    precisely the frozen baseline, while direct losses on logits give the
    output layers an unattenuated first-step gradient.  This deliberately
    avoids the old ``small gate × zero feature residual`` gradient path.
    """

    def __init__(self, neuron_width: int, hidden_dim: int, classes_num: int, kernel_size: int, delta_logit_cap: float) -> None:
        super().__init__()
        if neuron_width <= 0 or hidden_dim <= 0 or classes_num < 2:
            raise ValueError("neuron_width, hidden_dim and classes_num must be positive; classes_num must be at least two")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("adapter-kernel-size must be a positive odd integer")
        if delta_logit_cap <= 0:
            raise ValueError("delta-logit-cap must be positive")
        self.delta_logit_cap = float(delta_logit_cap)
        self.neuron_norm = nn.LayerNorm(int(neuron_width))
        self.input_projection = nn.Linear(int(neuron_width), int(hidden_dim))
        self.temporal = nn.Conv1d(
            int(hidden_dim), int(hidden_dim), kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2, groups=int(hidden_dim), bias=True,
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.binary_head = nn.Linear(int(hidden_dim), 1)
        self.class_head = nn.Linear(int(hidden_dim), int(classes_num))
        nn.init.zeros_(self.binary_head.weight)
        nn.init.zeros_(self.binary_head.bias)
        nn.init.zeros_(self.class_head.weight)
        nn.init.zeros_(self.class_head.bias)

    def forward(self, neuron: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if neuron.ndim != 3:
            raise ValueError(f"expected neuron [B,T,D], got {tuple(neuron.shape)}")
        normalized = self.neuron_norm(neuron)
        hidden = functional.gelu(self.input_projection(normalized))
        hidden = hidden + functional.gelu(self.temporal(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.output_norm(hidden)
        raw_binary, raw_class = self.binary_head(hidden), self.class_head(hidden)
        # The cap has unit derivative at zero, so it bounds late corrections
        # without weakening the direct first-step optimisation signal.
        binary = self.delta_logit_cap * torch.tanh(raw_binary / self.delta_logit_cap)
        classes = self.delta_logit_cap * torch.tanh(raw_class / self.delta_logit_cap)
        return binary, classes, normalized


class DirectScoreResidualVadCLIP(nn.Module):
    """Frozen VadCLIP plus direct selected-neuron corrections to both score heads."""

    def __init__(
        self,
        options,
        vadclip_root: str,
        device: str,
        neuron_width: int = 768,
        clip_dim: int = 512,
        adapter_hidden_dim: int = 256,
        adapter_kernel_size: int = 5,
        delta_logit_cap: float = 2.0,
    ) -> None:
        super().__init__()
        if int(clip_dim) != 512:
            raise ValueError("VadCLIP's final CLIP interface is fixed at 512D")
        add_vadclip_source(vadclip_root)
        from model import CLIPVAD

        self.neuron_width, self.clip_dim = int(neuron_width), int(clip_dim)
        self.base = CLIPVAD(
            options.classes_num, options.embed_dim, options.visual_length, self.clip_dim,
            options.visual_head, options.visual_layers, options.attn_window,
            options.prompt_prefix, options.prompt_postfix, device,
        )
        self.adapter = TemporalScoreAdapter(
            self.neuron_width, int(adapter_hidden_dim), int(options.classes_num),
            int(adapter_kernel_size), float(delta_logit_cap),
        )
        self._base_frozen = False

    def freeze_base(self) -> None:
        """Freeze all official VadCLIP tensors; only ``adapter`` remains trainable."""
        self.base.requires_grad_(False)
        self._base_frozen = True

    def train(self, mode: bool = True):
        """Keep official VadCLIP train/eval transitions despite frozen weights."""
        super().train(mode)
        return self

    def forward_with_details(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        neuron, clip = split_concat(visual, self.neuron_width, self.clip_dim)
        text_features, base_logits1, base_logits2 = self.base(clip, padding_mask, text, lengths)
        delta_logits1, delta_logits2, normalized_neuron = self.adapter(neuron)
        if base_logits1.shape != delta_logits1.shape or base_logits2.shape != delta_logits2.shape:
            raise RuntimeError(
                "score-adapter output does not match frozen VadCLIP heads: "
                f"binary={tuple(delta_logits1.shape)} vs {tuple(base_logits1.shape)}, "
                f"class={tuple(delta_logits2.shape)} vs {tuple(base_logits2.shape)}"
            )
        details = {
            "base_logits1": base_logits1,
            "base_logits2": base_logits2,
            "delta_logits1": delta_logits1,
            "delta_logits2": delta_logits2,
            "normalized_neuron": normalized_neuron,
        }
        return text_features, base_logits1 + delta_logits1, base_logits2 + delta_logits2, details

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        text_features, logits1, logits2, _details = self.forward_with_details(visual, padding_mask, text, lengths)
        return text_features, logits1, logits2


def build_score_residual_model(
    options,
    vadclip_root: str,
    device: str,
    contract: dict,
    adapter_hidden_dim: int,
    adapter_kernel_size: int,
    delta_logit_cap: float,
) -> DirectScoreResidualVadCLIP:
    """Build only when the chosen global-neuron concat contract is valid."""
    neuron_width = int(contract.get("neuron_width", 768))
    clip_dim = int(contract.get("clip_dim", 512))
    input_width = int(contract.get("input_width", contract.get("visual_width", neuron_width + clip_dim)))
    if input_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    return DirectScoreResidualVadCLIP(
        options, vadclip_root, device, neuron_width, clip_dim,
        adapter_hidden_dim, adapter_kernel_size, delta_logit_cap,
    )


def valid_snippet_mask(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Mask true snippets, never padded zeros, in every new loss/statistic."""
    positions = torch.arange(int(time_steps), device=lengths.device).unsqueeze(0)
    valid_lengths = lengths.to(torch.long).clamp(min=0, max=int(time_steps)).unsqueeze(1)
    return positions < valid_lengths


def score_residual_statistics(details: dict[str, torch.Tensor], lengths: torch.Tensor) -> dict[str, torch.Tensor]:
    """Measure correction magnitude and probability movement on valid snippets."""
    delta1, delta2 = details["delta_logits1"], details["delta_logits2"]
    valid = valid_snippet_mask(lengths, delta1.shape[1])
    valid_float = valid.to(delta1.dtype)
    base_prob2 = 1.0 - details["base_logits2"].softmax(dim=-1)[..., 0]
    enhanced_prob2 = 1.0 - (details["base_logits2"] + delta2).softmax(dim=-1)[..., 0]
    denominator = valid_float.sum().clamp_min(1.0)
    return {
        "delta1_abs_mean": (delta1.squeeze(-1).abs() * valid_float).sum() / denominator,
        "delta2_abs_mean": (delta2.abs().mean(dim=-1) * valid_float).sum() / denominator,
        "prob2_shift_abs_mean": ((enhanced_prob2 - base_prob2).abs() * valid_float).sum() / denominator,
    }


def feature_edge_delta_loss(details: dict[str, torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
    """Smooth only new score corrections across feature-consistent neighbours.

    Consecutive snippets with low selected-neuron cosine similarity receive a
    small weight, so an abrupt scene transition is not forced to have equal
    corrected scores.  The frozen baseline scores never appear in this term.
    """
    delta1, delta2 = details["delta_logits1"], details["delta_logits2"]
    if delta1.shape[1] < 2:
        return delta1.new_zeros(())
    valid = valid_snippet_mask(lengths, delta1.shape[1])
    pair_valid = valid[:, 1:] & valid[:, :-1]
    if not bool(pair_valid.any()):
        return delta1.new_zeros(())
    feature = functional.normalize(details["normalized_neuron"], dim=-1, eps=1e-6)
    similarity = (feature[:, 1:] * feature[:, :-1]).sum(dim=-1).clamp(min=0.0)
    binary_difference = functional.smooth_l1_loss(
        delta1[:, 1:], delta1[:, :-1], reduction="none"
    ).squeeze(-1)
    class_difference = functional.smooth_l1_loss(
        delta2[:, 1:], delta2[:, :-1], reduction="none"
    ).mean(dim=-1)
    weights = similarity * pair_valid.to(similarity.dtype)
    return (weights * 0.5 * (binary_difference + class_difference)).sum() / weights.sum().clamp_min(1.0)
