"""Repository-local checkpoint and import helpers for adaptive residual experiments."""
from __future__ import annotations

import sys
from pathlib import Path


INJECTION_DIR = Path(__file__).resolve().parents[1] / "vadclip_neuron_injection"


def add_injection_source() -> None:
    """Expose shared feature/evaluation helpers from this repository only."""
    source = str(INJECTION_DIR)
    if source not in sys.path:
        sys.path.append(source)


def state_dict_from_file(path: str) -> dict:
    """Load either a raw model state or a resumable checkpoint."""
    import torch

    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def initialize_adaptive_from_baseline(model, baseline_path: str) -> int:
    """Copy only frozen official VadCLIP tensors into a new adaptive wrapper."""
    source, target = state_dict_from_file(baseline_path), model.state_dict()
    missing, mismatched, copied = [], [], 0
    for name, tensor in target.items():
        if not name.startswith("base."):
            continue
        source_name = name.removeprefix("base.")
        candidate = source.get(source_name)
        if candidate is None:
            missing.append(source_name)
        elif tuple(candidate.shape) != tuple(tensor.shape):
            mismatched.append((source_name, tuple(candidate.shape), tuple(tensor.shape)))
        else:
            target[name] = candidate
            copied += 1
    if missing or mismatched:
        raise RuntimeError(
            "baseline model is incompatible with official 512D VadCLIP: "
            f"missing={missing[:4]}, shape_mismatch={mismatched[:2]}"
        )
    model.load_state_dict(target, strict=True)
    return copied
