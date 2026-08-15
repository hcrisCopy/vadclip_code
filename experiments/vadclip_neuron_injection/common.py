"""Shared, repository-local utilities for the VadCLIP neuron experiment.

The module deliberately contains no imports from the DSANet experiment.  It
accepts ordinary CSV/NumPy inputs from the shared ``../vad_data`` cache and
writes new VadCLIP experiment artifacts under independent ``../vadclip_data``
paths.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CHUNK_RE = re.compile(r"__(\d+)$")

UCF_TRAIN_LABELS = {
    "Normal": "normal", "Abuse": "abuse", "Arrest": "arrest", "Arson": "arson",
    "Assault": "assault", "Burglary": "burglary", "Explosion": "explosion",
    "Fighting": "fighting", "RoadAccidents": "roadAccidents", "Robbery": "robbery",
    "Shooting": "shooting", "Shoplifting": "shoplifting", "Stealing": "stealing",
    "Vandalism": "vandalism",
}

# These strings follow the official VadCLIP UCF test launcher exactly.
UCF_TEST_LABELS = {
    "Normal": "Normal", "Abuse": "Abuse", "Arrest": "Arrest", "Arson": "Arson",
    "Assault": "Assault", "Burglary": "Burglary", "Explosion": "Explosion",
    "Fighting": "Fighting", "RoadAccidents": "RoadAccidents", "Robbery": "Robbery",
    "Shooting": "Shooting", "Shoplifting": "Shoplifting", "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}

# These strings and order follow ``VadCLIP/src/xd_train.py`` exactly.
XD_LABELS = {
    "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
    "B5": "abuse", "B6": "car accident", "G": "explosion",
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def clean_dir(path: str | Path) -> Path:
    """Remove an explicitly requested experiment-output directory safely."""
    target = Path(path)
    if target in {Path("."), Path("..")} or str(target) in {"", "."} or target.name in {"", "."}:
        raise ValueError("--clean refuses to remove the current directory")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def feature_key(path: str) -> str:
    return Path(str(path)).stem


def base_key(path_or_key: str) -> str:
    return CHUNK_RE.sub("", feature_key(path_or_key))


def chunk_index(path_or_key: str) -> int:
    match = CHUNK_RE.search(feature_key(path_or_key))
    return int(match.group(1)) if match else 0


def is_normal_label(dataset: str, label: str) -> bool:
    dataset = dataset.lower()
    if dataset == "ucf":
        return str(label) == "Normal"
    if dataset == "xd":
        # XD labels may include multiple event codes (for example ``B1-B2``).
        # Official VadCLIP uses ``A`` exclusively for normal videos.
        return str(label).split("-")[0] == "A"
    raise ValueError(f"unsupported dataset={dataset!r}; expected 'ucf' or 'xd'")


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def grouped_rows(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = frame.copy()
    work["_base_key"] = work["path"].map(base_key)
    work["_chunk_index"] = work["path"].map(chunk_index)
    return {key: group.sort_values("_chunk_index") for key, group in work.groupby("_base_key")}


def load_hidden(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    artifact = np.load(path, allow_pickle=True)
    if not isinstance(artifact, np.lib.npyio.NpzFile) or "hidden" not in artifact.files:
        raise ValueError(f"{path}: expected an .npz hidden artifact containing key 'hidden'")
    hidden = np.asarray(artifact["hidden"], dtype=np.float32)
    metadata = {
        key: artifact[key].item() if artifact[key].shape == () else artifact[key]
        for key in artifact.files
        if key != "hidden"
    }
    return hidden, metadata


def load_clip_feature(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=True)
    if isinstance(artifact, np.lib.npyio.NpzFile):
        key = "features" if "features" in artifact.files else artifact.files[0]
        feature = artifact[key]
    else:
        feature = artifact
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError(f"{path}: expected [T,D] feature array, got {feature.shape}")
    if feature.shape[0] == 0:
        raise ValueError(f"{path}: feature sequence is empty")
    return feature


def resample_scores(scores: np.ndarray, target_len: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if len(scores) == target_len:
        return scores
    if len(scores) == 0:
        return np.zeros(target_len, dtype=np.float32)
    return np.interp(
        np.linspace(0.0, 1.0, target_len, dtype=np.float32),
        np.linspace(0.0, 1.0, len(scores), dtype=np.float32),
        scores,
    ).astype(np.float32)


def uniform_indices(length: int, count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("cannot sample an empty sequence")
    count = max(1, min(int(count), int(length)))
    return np.linspace(0, length - 1, count, dtype=np.int64)


def save_json(path: str | Path, content: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    import csv

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)


def process_train_feature(feature: np.ndarray, visual_length: int) -> tuple[np.ndarray, int]:
    """Match VadCLIP ``utils.tools.process_feat`` for UCF training."""
    feature = np.asarray(feature, dtype=np.float32)
    original_length = int(feature.shape[0])
    if original_length > visual_length:
        reduced = np.zeros((visual_length, feature.shape[1]), dtype=np.float32)
        boundaries = np.linspace(0, original_length, visual_length + 1, dtype=np.int32)
        for index in range(visual_length):
            left, right = boundaries[index], boundaries[index + 1]
            reduced[index] = feature[left:right].mean(axis=0) if left != right else feature[left]
        return reduced, visual_length
    if original_length < visual_length:
        feature = np.pad(feature, ((0, visual_length - original_length), (0, 0)), mode="constant")
    return feature.astype(np.float32), original_length


def process_test_feature(feature: np.ndarray, visual_length: int) -> tuple[np.ndarray, int]:
    """Match VadCLIP ``utils.tools.process_split``, including its final pad chunk."""
    feature = np.asarray(feature, dtype=np.float32)
    original_length = int(feature.shape[0])
    if original_length < visual_length:
        return np.pad(feature, ((0, visual_length - original_length), (0, 0)), mode="constant"), original_length
    split_num = int(original_length / visual_length) + 1
    chunks = []
    for index in range(split_num):
        part = feature[index * visual_length:index * visual_length + visual_length]
        if part.shape[0] < visual_length:
            part = np.pad(part, ((0, visual_length - part.shape[0]), (0, 0)), mode="constant")
        chunks.append(part.reshape(1, visual_length, feature.shape[1]))
    return np.concatenate(chunks, axis=0), original_length


def test_chunk_lengths(original_length: int, visual_length: int) -> np.ndarray:
    """Reproduce VadCLIP test's per-chunk valid-length construction."""
    remaining = int(original_length)
    lengths = []
    for index in range(int(original_length / visual_length) + 1):
        if index == 0 and original_length < visual_length:
            lengths.append(original_length)
        elif index == 0 and original_length > visual_length:
            lengths.append(visual_length)
            remaining -= visual_length
        elif remaining > visual_length:
            lengths.append(visual_length)
            remaining -= visual_length
        else:
            lengths.append(remaining)
    return np.asarray(lengths, dtype=np.int64)
