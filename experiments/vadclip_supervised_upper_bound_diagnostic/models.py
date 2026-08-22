"""Frozen VadCLIP with a class-preserving global-768 score correction."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn


def add_vadclip_source(vadclip_root: str) -> None:
    """Expose the unmodified official VadCLIP implementation."""
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def split_concat(visual: torch.Tensor, neuron_width: int, clip_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the [global-neuron | CLIP] concat-feature contract."""
    expected = int(neuron_width) + int(clip_dim)
    if visual.ndim != 3 or visual.shape[-1] != expected:
        raise ValueError(f"expected [B,T,{expected}] [neuron|clip], got {tuple(visual.shape)}")
    return visual[..., :neuron_width], visual[..., neuron_width:]


class ScalarTemporalAdapter(nn.Module):
    """Map selected neurons to one bounded anomaly-logit correction per snippet.

    The output head is zero initialised, so the diagnostic begins from the
    supplied frozen baseline exactly.  Unlike two independent class heads, a
    scalar correction cannot alter the ordering among anomaly categories.
    """

    def __init__(self, neuron_width: int, hidden_dim: int, kernel_size: int, delta_logit_cap: float) -> None:
        super().__init__()
        if neuron_width <= 0 or hidden_dim <= 0:
            raise ValueError("neuron_width and hidden_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if delta_logit_cap <= 0:
            raise ValueError("delta_logit_cap must be positive")
        self.delta_logit_cap = float(delta_logit_cap)
        self.neuron_norm = nn.LayerNorm(int(neuron_width))
        self.input_projection = nn.Linear(int(neuron_width), int(hidden_dim))
        self.temporal = nn.Conv1d(
            int(hidden_dim), int(hidden_dim), kernel_size=int(kernel_size),
            padding=int(kernel_size) // 2, groups=int(hidden_dim), bias=True,
        )
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, neuron: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if neuron.ndim != 3:
            raise ValueError(f"expected neuron [B,T,D], got {tuple(neuron.shape)}")
        normalized = self.neuron_norm(neuron)
        hidden = functional.gelu(self.input_projection(normalized))
        hidden = hidden + functional.gelu(self.temporal(hidden.transpose(1, 2)).transpose(1, 2))
        hidden = self.output_norm(hidden)
        raw_delta = self.output(hidden)
        delta = self.delta_logit_cap * torch.tanh(raw_delta / self.delta_logit_cap)
        return delta, normalized


class ClassPreservingScoreVadCLIP(nn.Module):
    """Frozen VadCLIP plus a scalar anomaly-vs-normal correction.

    ``delta > 0`` increases the binary anomaly logit and increases every
    anomaly-language logit by the same amount relative to the normal logit.
    Thus the diagnostic can change temporal eventness but cannot invent a
    different anomaly category ranking.
    """

    def __init__(
        self,
        options,
        vadclip_root: str,
        device: str,
        neuron_width: int,
        clip_dim: int,
        adapter_hidden_dim: int,
        adapter_kernel_size: int,
        delta_logit_cap: float,
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
        self.adapter = ScalarTemporalAdapter(
            self.neuron_width, int(adapter_hidden_dim), int(adapter_kernel_size), float(delta_logit_cap)
        )

    def freeze_base(self) -> None:
        """Keep all official VadCLIP parameters fixed for this diagnosis."""
        self.base.requires_grad_(False)

    def forward_with_details(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        neuron, clip = split_concat(visual, self.neuron_width, self.clip_dim)
        text_features, base_logits1, base_logits2 = self.base(clip, padding_mask, text, lengths)
        delta, normalized_neuron = self.adapter(neuron)
        if base_logits1.shape != delta.shape:
            raise RuntimeError(f"binary correction shape mismatch: {tuple(base_logits1.shape)} vs {tuple(delta.shape)}")
        if base_logits2.shape[-1] < 2:
            raise RuntimeError("VadCLIP must expose normal plus at least one anomaly class")
        class_offset = torch.cat((-0.5 * delta, 0.5 * delta.expand_as(base_logits2[..., 1:])), dim=-1)
        enhanced_logits1 = base_logits1 + delta
        enhanced_logits2 = base_logits2 + class_offset
        return text_features, enhanced_logits1, enhanced_logits2, {
            "base_logits1": base_logits1,
            "base_logits2": base_logits2,
            "delta": delta,
            "normalized_neuron": normalized_neuron,
        }

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        text_features, logits1, logits2, _details = self.forward_with_details(visual, padding_mask, text, lengths)
        return text_features, logits1, logits2


def build_model(
    options,
    vadclip_root: str,
    device: str,
    contract: dict,
    adapter_hidden_dim: int,
    adapter_kernel_size: int,
    delta_logit_cap: float,
) -> ClassPreservingScoreVadCLIP:
    """Build only when the selected-neuron concat metadata is self-consistent."""
    neuron_width = int(contract.get("neuron_width", 768))
    clip_dim = int(contract.get("clip_dim", 512))
    input_width = int(contract.get("input_width", contract.get("visual_width", neuron_width + clip_dim)))
    if input_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    return ClassPreservingScoreVadCLIP(
        options, vadclip_root, device, neuron_width, clip_dim,
        adapter_hidden_dim, adapter_kernel_size, delta_logit_cap,
    )


def valid_snippet_mask(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Mask real snippets and exclude right-padding from every loss."""
    positions = torch.arange(int(time_steps), device=lengths.device).unsqueeze(0)
    valid_lengths = lengths.to(torch.long).clamp(min=0, max=int(time_steps)).unsqueeze(1)
    return positions < valid_lengths


def correction_statistics(details: dict[str, torch.Tensor], lengths: torch.Tensor) -> dict[str, torch.Tensor]:
    """Report whether the supervised diagnostic actually moves the baseline."""
    delta = details["delta"].squeeze(-1)
    valid = valid_snippet_mask(lengths, delta.shape[1]).to(delta.dtype)
    denominator = valid.sum().clamp_min(1.0)
    base_prob2 = 1.0 - details["base_logits2"].softmax(dim=-1)[..., 0]
    adjusted = details["base_logits2"].clone()
    adjusted[..., 0] -= 0.5 * details["delta"].squeeze(-1)
    adjusted[..., 1:] += 0.5 * details["delta"]
    enhanced_prob2 = 1.0 - adjusted.softmax(dim=-1)[..., 0]
    return {
        "delta_abs_mean": (delta.abs() * valid).sum() / denominator,
        "prob2_shift_abs_mean": ((enhanced_prob2 - base_prob2).abs() * valid).sum() / denominator,
    }
