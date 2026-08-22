"""Small, repository-local helpers for online selected-neuron adaptation."""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def add_vadclip_source(vadclip_root: str) -> None:
    """Make the unmodified official VadCLIP source importable."""
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def clean_dir(path: str | Path) -> Path:
    """Delete only the explicit output directory passed by ``--clean``."""
    target = Path(path)
    if target in {Path("."), Path(".."), Path("/")} or target.name in {"", "."}:
        raise ValueError("--clean refuses to remove a broad directory")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def feature_key(path: str) -> str:
    return Path(str(path)).stem


def base_key(path: str) -> str:
    """Match the existing concat builder's ``video__chunk`` key convention."""
    stem = feature_key(path)
    head, marker, tail = stem.rpartition("__")
    return head if marker and tail.isdigit() else stem


def resolve_existing_path(value: str | Path, anchor: str | Path | None = None) -> Path:
    """Resolve an absolute path or a project-relative manifest path safely."""
    original = Path(str(value))
    candidates = [original]
    if anchor is not None and not original.is_absolute():
        anchor_path = Path(anchor)
        candidates.extend([anchor_path / original, anchor_path.parent / original])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve path {value!r}; tried: {joined}")


def load_feature(path: str | Path) -> np.ndarray:
    """Load a standard VadCLIP [T,512] NumPy feature file."""
    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        key = "features" if "features" in value.files else value.files[0]
        value = value[key]
    feature = np.asarray(value, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] == 0:
        raise ValueError(f"{path}: expected non-empty [T,D] feature, got {feature.shape}")
    return feature


def feature_length(path: str | Path) -> int:
    """Read only the temporal length of a source 512D feature when possible."""
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        key = "features" if "features" in value.files else value.files[0]
        value = value[key]
    if value.ndim != 2 or value.shape[0] <= 0:
        raise ValueError(f"{path}: expected non-empty [T,D] feature, got {value.shape}")
    return int(value.shape[0])


def load_selected_dims(neuron_json: str | Path, expected_layers: int = 12) -> list[np.ndarray]:
    """Read the global-768 selection as one sorted dimension list per CLIP block."""
    config = load_json(neuron_json)
    if str(config.get("token_pool", "cls")) != "cls":
        raise ValueError("online Adapter currently supports only CLS-token neuron selections")
    selected = config.get("selected")
    if not isinstance(selected, list):
        raise ValueError(f"{neuron_json}: missing selected-neuron list")
    output: list[np.ndarray | None] = [None] * expected_layers
    for item in selected:
        layer = int(item["layer_index"])
        dims = np.asarray(item["dims"], dtype=np.int64)
        if not 0 <= layer < expected_layers:
            raise ValueError(f"selected layer {layer} is outside [0,{expected_layers})")
        if dims.ndim != 1 or len(dims) == 0 or np.any(dims < 0) or np.any(dims >= 768):
            raise ValueError(f"layer {layer}: invalid selected dimensions")
        if len(np.unique(dims)) != len(dims):
            raise ValueError(f"layer {layer}: duplicate selected dimensions")
        if output[layer] is not None:
            raise ValueError(f"duplicate entry for layer {layer}")
        output[layer] = np.sort(dims)
    if any(dims is None for dims in output):
        absent = [index for index, dims in enumerate(output) if dims is None]
        raise ValueError(f"selection must contain every CLIP layer; absent={absent}")
    result = [dims for dims in output if dims is not None]
    if sum(len(dims) for dims in result) != 768:
        raise ValueError(f"expected global-768 selection, got {sum(len(dims) for dims in result)} dimensions")
    return result


def baseline_options(vadclip_root: str):
    add_vadclip_source(vadclip_root)
    import xd_option

    return xd_option.parser.parse_args([])


def state_dict_from_file(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a state dict")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state
