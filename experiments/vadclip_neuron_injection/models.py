"""Residual injection model that leaves the official VadCLIP baseline untouched."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def add_vadclip_source(vadclip_root: str) -> None:
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def split_concat(visual: torch.Tensor, neuron_width: int, clip_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    expected = neuron_width + clip_dim
    if visual.ndim != 3 or visual.shape[-1] != expected:
        raise ValueError(f"expected [B,T,{expected}] [neuron|clip], got {tuple(visual.shape)}")
    return visual[..., :neuron_width], visual[..., neuron_width:]


class ResidualInjectionVadCLIP(nn.Module):
    """Add a small, learned 768D-neuron correction before frozen VadCLIP.

    The residual MLP's final linear layer is zero initialized, so the initial
    forward pass is exactly the supplied 512D VadCLIP baseline.  Training only
    updates the LayerNorm, MLP and scalar sigmoid gate.
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
        final_linear = next(layer for layer in reversed(self.neuron_to_clip) if isinstance(layer, nn.Linear))
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        self.gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.base = CLIPVAD(
            options.classes_num, options.embed_dim, options.visual_length, self.clip_dim,
            options.visual_head, options.visual_layers, options.attn_window,
            options.prompt_prefix, options.prompt_postfix, device,
        )
        self._base_frozen = False

    def residual_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()
        self._base_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._base_frozen:
            self.base.eval()
        return self

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        neuron, clip = split_concat(visual, self.neuron_width, self.clip_dim)
        correction = self.neuron_to_clip(self.neuron_norm(neuron))
        enhanced_clip = clip + self.residual_gate().to(clip.dtype) * correction.to(clip.dtype)
        return self.base(enhanced_clip, padding_mask, text, lengths)


def build_residual_model(options, vadclip_root: str, device: str, contract: dict, residual_hidden_dim: int, residual_depth: int) -> ResidualInjectionVadCLIP:
    neuron_width = int(contract.get("neuron_width", 768))
    clip_dim = int(contract.get("clip_dim", 512))
    input_width = int(contract.get("input_width", contract.get("visual_width", neuron_width + clip_dim)))
    if input_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    return ResidualInjectionVadCLIP(
        options, vadclip_root, device, neuron_width, clip_dim, residual_hidden_dim, residual_depth
    )
