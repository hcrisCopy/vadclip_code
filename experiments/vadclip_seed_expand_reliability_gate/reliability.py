"""Label-free reliability maps shared by seed selection and XD evaluation.

The module never reads test annotations.  A reliability value is high only
when a frozen baseline score is unusual relative to *training normal* scores
and its CLIP hidden state resembles the video's high-score seed prototype.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import base_key, is_normal_label, load_hidden, read_csv, resample_scores, uniform_indices


@dataclass(frozen=True)
class ReliabilityConfig:
    """Parameters fixed before evaluating or training a residual model."""

    seed_top_p: float
    expand_top_p: float
    normal_score_quantile: float
    score_temperature: float
    sigma_min: float
    normal_score_threshold: float


def manifest_map(path: str) -> dict[str, str]:
    """Read one hidden artifact path per video key with consistency checks."""
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns: {sorted(missing)}")
    result: dict[str, str] = {}
    for _, row in frame.iterrows():
        key, hidden_path = str(row["key"]), str(row["hidden_path"])
        if key in result and result[key] != hidden_path:
            raise ValueError(f"{path}: duplicate key {key!r} has different hidden paths")
        result[key] = hidden_path
    return result


def manifest_token_pool(path: str) -> str:
    """Validate that all reused hidden features use the same token pooling."""
    frame = pd.read_csv(path)
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame.columns else {"cls"}
    if pools - {"cls", "patch_mean"} or len(pools) != 1:
        raise ValueError(f"{path}: expected one valid token_pool, got {sorted(pools)}")
    return next(iter(pools))


def labels_by_key(path: str) -> dict[str, str]:
    """Read a stable video-level label map from a local path,label CSV."""
    result: dict[str, str] = {}
    for _, row in read_csv(path).iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in result and result[key] != label:
            raise ValueError(f"{path}: inconsistent labels for video {key!r}")
        result[key] = label
    return result


def pseudo_score_paths(path: str) -> dict[str, tuple[str, str]]:
    """Map video keys to frozen classifier-probability score arrays."""
    frame = pd.read_csv(path)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing pseudo-score columns: {sorted(missing)}")
    result: dict[str, tuple[str, str]] = {}
    for _, row in frame.iterrows():
        key, value = str(row["key"]), (str(row["label"]), str(row["score_path"]))
        if key in result:
            raise ValueError(f"{path}: duplicate pseudo-score key {key!r}")
        result[key] = value
    return result


def top_indices(scores: np.ndarray, top_p: float) -> np.ndarray:
    """Return the deterministic top fraction used by the original selector."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError(f"need at least two scores, got {values.size}")
    count = min(max(1, int(np.ceil(top_p * values.size))), values.size // 2)
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def bounded_top_indices(scores: np.ndarray, top_p: float) -> np.ndarray:
    """Choose an expanded candidate set, while retaining at least one element."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot select from an empty reliability map")
    count = max(1, min(int(np.ceil(top_p * values.size)), values.size))
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def collect_normal_hidden_stats(hidden_paths: list[str], limit_per_video: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate per-neuron normal mean/std with an equal cap per video."""
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    for hidden_path in tqdm(hidden_paths, desc="pure-normal hidden statistics", unit="video"):
        hidden, _metadata = load_hidden(hidden_path)
        if hidden.ndim != 3 or hidden.shape[0] == 0:
            raise ValueError(f"{hidden_path}: expected non-empty [T,L,D], got {hidden.shape}")
        for snippet in hidden[uniform_indices(hidden.shape[0], min(limit_per_video, hidden.shape[0]))]:
            if mean is None:
                mean, m2 = np.zeros_like(snippet, dtype=np.float64), np.zeros_like(snippet, dtype=np.float64)
            elif snippet.shape != mean.shape:
                raise ValueError(f"{hidden_path}: layer/dimension shape differs from prior normal videos")
            count += 1
            delta = snippet - mean
            mean += delta / count
            m2 += delta * (snippet - mean)
    if count < 2 or mean is None or m2 is None:
        raise RuntimeError("need at least two pure-normal hidden snippets")
    std = np.sqrt(np.maximum(m2 / (count - 1), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), count


def collect_normal_score_threshold(
    dataset: str,
    source_train_csv: str,
    pseudo_csv: str,
    quantile: float,
    limit_per_video: int,
) -> tuple[float, int, int]:
    """Calibrate the frozen score against pure-normal training videos only."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("normal-score-quantile must be in (0.5, 1.0)")
    labels = labels_by_key(source_train_csv)
    score_paths = pseudo_score_paths(pseudo_csv)
    samples: list[np.ndarray] = []
    missing = 0
    for key, label in sorted(labels.items()):
        if not is_normal_label(dataset, label):
            continue
        item = score_paths.get(key)
        if item is None:
            missing += 1
            continue
        pseudo_label, score_path = item
        if pseudo_label != label:
            raise ValueError(f"{key}: pseudo score label {pseudo_label!r} differs from source label {label!r}")
        scores = np.asarray(np.load(score_path, allow_pickle=False), dtype=np.float32).reshape(-1)
        if scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError(f"{score_path}: expected finite non-empty score array")
        samples.append(scores[uniform_indices(scores.size, min(limit_per_video, scores.size))])
    if not samples:
        raise RuntimeError("no usable pure-normal pseudo scores were found")
    values = np.concatenate(samples)
    return float(np.quantile(values, quantile)), int(values.size), int(missing)


def build_config(
    dataset: str,
    source_train_csv: str,
    pseudo_csv: str,
    seed_top_p: float,
    expand_top_p: float,
    normal_score_quantile: float,
    score_temperature: float,
    sigma_min: float,
    normal_score_snippets_per_video: int,
) -> tuple[ReliabilityConfig, dict[str, int]]:
    """Build a fully label-free configuration from training-only artifacts."""
    if not 0.0 < seed_top_p <= 0.5 or not seed_top_p <= expand_top_p <= 0.5:
        raise ValueError("seed-top-p must be in (0, .5] and expand-top-p must be in [seed-top-p, .5]")
    if score_temperature <= 0.0 or sigma_min <= 0.0 or normal_score_snippets_per_video <= 0:
        raise ValueError("score-temperature, sigma-min, and normal score samples must be positive")
    threshold, samples, missing = collect_normal_score_threshold(
        dataset, source_train_csv, pseudo_csv, normal_score_quantile, normal_score_snippets_per_video
    )
    return ReliabilityConfig(
        seed_top_p=float(seed_top_p),
        expand_top_p=float(expand_top_p),
        normal_score_quantile=float(normal_score_quantile),
        score_temperature=float(score_temperature),
        sigma_min=float(sigma_min),
        normal_score_threshold=float(threshold),
    ), {"normal_score_samples": samples, "normal_videos_missing_pseudo_scores": missing}


def reliability_map(
    hidden: np.ndarray,
    baseline_scores: np.ndarray,
    normal_mean: np.ndarray,
    normal_std: np.ndarray,
    config: ReliabilityConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return q, aligned scores, seed indices, and seed-similarity values.

    The score factor is an absolute normal-calibrated probability, not a
    within-video rank.  The semantic factor is cosine similarity to the
    video-specific top-score seed prototype.  Their product prevents either
    signal alone from broadly activating the residual on normal content.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[0] == 0 or hidden.shape[1:] != normal_mean.shape:
        raise ValueError(f"hidden shape {hidden.shape} does not match normal statistics {normal_mean.shape}")
    if normal_std.shape != normal_mean.shape:
        raise ValueError("normal mean/std shapes differ")
    aligned = resample_scores(baseline_scores, hidden.shape[0])
    seeds = top_indices(aligned, config.seed_top_p)
    z_hidden = (hidden - normal_mean) / (normal_std + config.sigma_min)
    vectors = z_hidden.reshape(z_hidden.shape[0], -1).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    prototype = vectors[seeds].mean(axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
    similarity = np.clip(vectors @ prototype, 0.0, 1.0).astype(np.float32)
    score_confidence = 1.0 / (1.0 + np.exp(-np.clip(
        (aligned - config.normal_score_threshold) / config.score_temperature, -50.0, 50.0
    )))
    reliability = np.clip(score_confidence * similarity, 0.0, 1.0).astype(np.float32)
    return reliability, aligned.astype(np.float32), seeds, similarity


def config_as_dict(config: ReliabilityConfig) -> dict[str, float]:
    """Serialize only the parameters defining the label-free reliability map."""
    return {
        "seed_top_p": config.seed_top_p,
        "expand_top_p": config.expand_top_p,
        "normal_score_quantile": config.normal_score_quantile,
        "score_temperature": config.score_temperature,
        "sigma_min": config.sigma_min,
        "normal_score_threshold": config.normal_score_threshold,
        "reliability_definition": "sigmoid((classifier_prob1-normal_training_quantile)/temperature) * max(cosine(z_hidden, top_score_seed_prototype), 0)",
    }
