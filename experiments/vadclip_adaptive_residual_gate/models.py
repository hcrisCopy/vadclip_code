"""Per-snippet gated neuron residual module; the official VadCLIP code remains untouched."""
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


class AdaptiveResidualGateVadCLIP(nn.Module):
    """Inject a neuron residual with a separately learned gate for every snippet.

    The residual output and gate-predictor weight are zero initialized.  The
    gate bias is -4, hence a newly constructed wrapper is exactly the supplied
    VadCLIP baseline: the correction is zero before the first optimiser step.
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
        if residual_hidden_dim <= 0 or residual_depth < 1:
            raise ValueError("residual-hidden-dim and residual-depth must be positive")
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
        final_residual = next(layer for layer in reversed(self.neuron_to_clip) if isinstance(layer, nn.Linear))
        nn.init.zeros_(final_residual.weight)
        nn.init.zeros_(final_residual.bias)
        # A linear gate is intentionally small: unlike a global scalar, it can
        # differ across snippets without adding a second large feature encoder.
        self.snippet_gate = nn.Linear(self.neuron_width, 1)
        nn.init.zeros_(self.snippet_gate.weight)
        nn.init.constant_(self.snippet_gate.bias, -4.0)
        self.base = CLIPVAD(
            options.classes_num, options.embed_dim, options.visual_length, self.clip_dim,
            options.visual_head, options.visual_layers, options.attn_window,
            options.prompt_prefix, options.prompt_postfix, device,
        )
        self._base_frozen = False

    def residual_components(self, visual: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the differentiable adaptive correction before calling VadCLIP."""
        neuron, clip = split_concat(visual, self.neuron_width, self.clip_dim)
        normalized_neuron = self.neuron_norm(neuron)
        correction = self.neuron_to_clip(normalized_neuron)
        gate = torch.sigmoid(self.snippet_gate(normalized_neuron))
        delta = gate.to(clip.dtype) * correction.to(clip.dtype)
        return {"clip": clip, "correction": correction, "gate": gate, "delta": delta}

    def freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self._base_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        # Follow official VadCLIP train/eval transitions while base parameters
        # remain frozen, exactly as the original global-768 residual wrapper.
        return self

    def forward(self, visual: torch.Tensor, padding_mask, text: list[str], lengths: torch.Tensor, return_details: bool = False):
        details = self.residual_components(visual)
        output = self.base(details["clip"] + details["delta"], padding_mask, text, lengths)
        return (*output, details) if return_details else output


def build_adaptive_model(
    options, vadclip_root: str, device: str, contract: dict, residual_hidden_dim: int, residual_depth: int
) -> AdaptiveResidualGateVadCLIP:
    """Construct from the same concat-width contract as global-768."""
    neuron_width = int(contract.get("neuron_width", 768))
    clip_dim = int(contract.get("clip_dim", 512))
    input_width = int(contract.get("input_width", contract.get("visual_width", neuron_width + clip_dim)))
    if input_width != neuron_width + clip_dim:
        raise ValueError("selected_neurons.json has an invalid concat width contract")
    return AdaptiveResidualGateVadCLIP(
        options, vadclip_root, device, neuron_width, clip_dim, residual_hidden_dim, residual_depth
    )


def valid_snippet_mask(lengths: torch.Tensor, time_steps: int) -> torch.Tensor:
    """Mask true snippets only; padded train positions never receive new losses."""
    positions = torch.arange(time_steps, device=lengths.device).unsqueeze(0)
    valid_lengths = lengths.to(torch.long).clamp(min=0, max=time_steps).unsqueeze(1)
    return positions < valid_lengths


def adaptive_regularization(
    details: dict[str, torch.Tensor],
    lengths: torch.Tensor,
    ratio_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return bounded-amplitude and temporal-variation terms on valid snippets.

    Amplitude is zero while ``||delta|| / ||clip||`` is below ``ratio_cap``.
    The temporal term is normalized by adjacent CLIP norms so it does not
    depend on the baseline feature scale.
    """
    if ratio_cap <= 0.0:
        raise ValueError("ratio_cap must be positive")
    clip, delta = details["clip"], details["delta"]
    valid = valid_snippet_mask(lengths, clip.shape[1])
    clip_norm = clip.norm(dim=-1).clamp_min(1e-8)
    ratio = delta.norm(dim=-1) / clip_norm
    excess = torch.relu(ratio / float(ratio_cap) - 1.0)
    amplitude = (excess.square() * valid).sum() / valid.sum().clamp_min(1)
    if clip.shape[1] < 2:
        temporal = torch.zeros((), dtype=clip.dtype, device=clip.device)
    else:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        scale = (0.5 * (clip_norm[:, 1:] + clip_norm[:, :-1])).clamp_min(1e-8)
        temporal_ratio = (delta[:, 1:] - delta[:, :-1]).norm(dim=-1) / scale
        temporal = (temporal_ratio.square() * pair_valid).sum() / pair_valid.sum().clamp_min(1)
    statistics = {
        "gate_mean": (details["gate"].squeeze(-1) * valid).sum() / valid.sum().clamp_min(1),
        "ratio_mean": (ratio * valid).sum() / valid.sum().clamp_min(1),
        "ratio_p95": torch.quantile(ratio.masked_select(valid), 0.95) if bool(valid.any()) else torch.zeros_like(amplitude),
    }
    return amplitude, temporal, statistics
