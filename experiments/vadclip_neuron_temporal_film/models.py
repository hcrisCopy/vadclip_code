"""Frozen-VadCLIP feature enhancement conditioned on selected neurons.

M1 performs per-snippet feature-wise linear modulation (FiLM): selected
global-768 CLIP neurons determine which channels of the original 512D feature
should contribute to a residual.  M2 adds a small multi-scale temporal mixer
on the neuron condition before producing that modulation.  Neither variant
changes a file in ``VadCLIP/`` or updates any baseline parameter.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


def add_vadclip_source(vadclip_root: str) -> None:
    """Make the untouched official VadCLIP ``src`` modules importable."""
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def split_concat(
    visual: torch.Tensor, neuron_width: int, clip_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the existing ``[selected neurons | original CLIP]`` feature."""
    expected = neuron_width + clip_dim
    if visual.ndim != 3 or visual.shape[-1] != expected:
        raise ValueError(
            f"expected [B,T,{expected}] [neuron|clip], got {tuple(visual.shape)}"
        )
    return visual[..., :neuron_width], visual[..., neuron_width:]


def valid_time_mask(lengths: torch.Tensor | None, steps: int, dtype: torch.dtype) -> torch.Tensor | None:
    """Return ``[B,T,1]`` valid-token mask for padded train/test chunks."""
    if lengths is None:
        return None
    clipped = lengths.to(device=lengths.device, dtype=torch.long).clamp(min=0, max=steps)
    positions = torch.arange(steps, device=lengths.device).unsqueeze(0)
    return (positions < clipped.unsqueeze(1)).unsqueeze(-1).to(dtype=dtype)


class MultiScaleTemporalMixer(nn.Module):
    """Two inexpensive depth-wise temporal paths used only by M2.

    The input and output are ``[B,T,H]``.  Convolutions are depth-wise, so
    temporal context is added without a second large temporal transformer or
    a dependence on raw video/online CLIP features.
    """

    def __init__(self, hidden_dim: int, dilations: Iterable[int]) -> None:
        super().__init__()
        checked = tuple(int(value) for value in dilations)
        if not checked or any(value < 1 for value in checked):
            raise ValueError("temporal dilations must contain positive integers")
        self.paths = nn.ModuleList([
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                bias=False,
            )
            for dilation in checked
        ])
        self.mix = nn.Linear(hidden_dim * len(self.paths), hidden_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, condition: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is not None:
            condition = condition * valid_mask
        channels_first = condition.transpose(1, 2)
        paths = [self.activation(path(channels_first)).transpose(1, 2) for path in self.paths]
        context = condition + self.mix(torch.cat(paths, dim=-1))
        return context if valid_mask is None else context * valid_mask


class NeuronConditionedFiLMVadCLIP(nn.Module):
    """Inject selected-neuron information as an identity-start residual.

    The last projection is zero-initialised.  Consequently both M1 and M2
    produce exactly the original cached 512D feature before their first
    optimiser step; unlike the old scalar sigmoid gate, there is no permanent
    low-amplitude bottleneck on the learned residual.
    """

    def __init__(
        self,
        options,
        vadclip_root: str,
        device: str,
        neuron_width: int = 768,
        clip_dim: int = 512,
        condition_hidden_dim: int = 256,
        residual_hidden_dim: int = 512,
        module_variant: str = "m1",
        temporal_dilations: Iterable[int] = (1, 2),
    ) -> None:
        super().__init__()
        if clip_dim != 512:
            raise ValueError("VadCLIP's final CLIP interface is fixed at 512D")
        if condition_hidden_dim <= 0 or residual_hidden_dim <= 0:
            raise ValueError("condition-hidden-dim and residual-hidden-dim must be positive")
        if module_variant not in {"m1", "m2"}:
            raise ValueError("module_variant must be 'm1' or 'm2'")

        add_vadclip_source(vadclip_root)
        from model import CLIPVAD

        self.neuron_width = int(neuron_width)
        self.clip_dim = int(clip_dim)
        self.module_variant = module_variant
        self.neuron_norm = nn.LayerNorm(self.neuron_width)
        self.clip_norm = nn.LayerNorm(self.clip_dim)
        self.neuron_project = nn.Sequential(
            nn.Linear(self.neuron_width, int(condition_hidden_dim)),
            nn.GELU(),
        )
        self.temporal_mixer = (
            MultiScaleTemporalMixer(int(condition_hidden_dim), temporal_dilations)
            if module_variant == "m2" else nn.Identity()
        )
        # ``log1p(||condition||)`` supplies magnitude information without
        # multiplying the original feature by an unstable raw norm.
        condition_width = int(condition_hidden_dim) + 1
        self.modulation = nn.Sequential(
            nn.Linear(condition_width, int(condition_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(condition_hidden_dim), self.clip_dim * 2),
        )
        self.residual_body = nn.Sequential(
            nn.Linear(self.clip_dim, int(residual_hidden_dim)),
            nn.GELU(),
        )
        self.residual_out = nn.Linear(int(residual_hidden_dim), self.clip_dim)
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)
        self.base = CLIPVAD(
            options.classes_num,
            options.embed_dim,
            options.visual_length,
            self.clip_dim,
            options.visual_head,
            options.visual_layers,
            options.attn_window,
            options.prompt_prefix,
            options.prompt_postfix,
            device,
        )
        self._base_frozen = False
        self._last_statistics: dict[str, float] = {}

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self._base_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep the official train/eval state transitions.  ``requires_grad``
        # is what freezes the baseline, rather than forcing it to eval mode.
        return self

    def enhancement(self, neuron: torch.Tensor, clip: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor:
        """Return enhanced 512D features while recording detached diagnostics."""
        mask = valid_time_mask(lengths, neuron.shape[1], neuron.dtype)
        condition = self.neuron_project(self.neuron_norm(neuron))
        if mask is not None:
            condition = condition * mask
        if self.module_variant == "m2":
            condition = self.temporal_mixer(condition, mask)
        magnitude = torch.log1p(torch.linalg.vector_norm(condition, dim=-1, keepdim=True))
        gamma_beta = self.modulation(torch.cat([condition, magnitude], dim=-1))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # Bounded FiLM avoids arbitrarily rescaling frozen CLIP features.
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)
        modulated = (1.0 + gamma) * self.clip_norm(clip) + beta
        delta = self.residual_out(self.residual_body(modulated)).to(dtype=clip.dtype)
        if mask is not None:
            delta = delta * mask.to(dtype=delta.dtype)
        with torch.no_grad():
            valid = mask if mask is not None else torch.ones_like(delta[..., :1])
            denominator = valid.sum().clamp_min(1.0)
            self._last_statistics = {
                "residual_l2": float((torch.linalg.vector_norm(delta.float(), dim=-1, keepdim=True) * valid).sum() / denominator),
                "clip_l2": float((torch.linalg.vector_norm(clip.float(), dim=-1, keepdim=True) * valid).sum() / denominator),
                "gamma_abs": float((gamma.float().abs().mean(dim=-1, keepdim=True) * valid).sum() / denominator),
            }
        return clip + delta

    def residual_statistics(self) -> dict[str, float]:
        """Diagnostics shown in the training progress bar after each forward."""
        result = dict(self._last_statistics)
        clip_l2 = result.get("clip_l2", 0.0)
        result["residual_ratio"] = result.get("residual_l2", 0.0) / max(clip_l2, 1e-12)
        return result

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor):
        neuron, clip = split_concat(visual, self.neuron_width, self.clip_dim)
        enhanced_clip = self.enhancement(neuron, clip, lengths)
        return self.base(enhanced_clip, padding_mask, text, lengths)


def build_film_model(
    options,
    vadclip_root: str,
    device: str,
    contract: dict,
    module_variant: str,
    condition_hidden_dim: int,
    residual_hidden_dim: int,
    temporal_dilations: Iterable[int],
) -> NeuronConditionedFiLMVadCLIP:
    """Validate the persisted concat contract and build M1 or M2."""
    neuron_width = int(contract.get("neuron_width", 768))
    clip_dim = int(contract.get("clip_dim", 512))
    input_width = int(contract.get("input_width", contract.get("visual_width", neuron_width + clip_dim)))
    if input_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    return NeuronConditionedFiLMVadCLIP(
        options=options,
        vadclip_root=vadclip_root,
        device=device,
        neuron_width=neuron_width,
        clip_dim=clip_dim,
        condition_hidden_dim=condition_hidden_dim,
        residual_hidden_dim=residual_hidden_dim,
        module_variant=module_variant,
        temporal_dilations=temporal_dilations,
    )
