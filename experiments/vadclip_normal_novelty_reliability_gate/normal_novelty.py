"""Training-normal calibration and label-free normal-novelty reliability maps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import base_key, is_normal_label, load_hidden, read_csv, resample_scores, uniform_indices


@dataclass(frozen=True)
class NoveltyCalibration:
    """All scalar thresholds are fitted from normal training videos only."""

    score_threshold: float
    score_temperature: float
    novelty_threshold: float
    novelty_temperature: float
    seed_top_p: float
    expand_top_p: float
    sigma_min: float


def manifest_map(path: str) -> dict[str, str]:
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns: {sorted(missing)}")
    mapping: dict[str, str] = {}
    for _, row in frame.iterrows():
        key, hidden_path = str(row["key"]), str(row["hidden_path"])
        if key in mapping and mapping[key] != hidden_path:
            raise ValueError(f"{path}: duplicate key {key!r} has different hidden paths")
        mapping[key] = hidden_path
    return mapping


def labels_by_key(path: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for _, row in read_csv(path).iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"{path}: inconsistent labels for {key!r}")
        labels[key] = label
    return labels


def pseudo_score_paths(path: str) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(path)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing pseudo-score columns: {sorted(missing)}")
    result: dict[str, tuple[str, str]] = {}
    for _, row in frame.iterrows():
        key = str(row["key"])
        if key in result:
            raise ValueError(f"{path}: duplicate pseudo-score key {key!r}")
        result[key] = (str(row["label"]), str(row["score_path"]))
    return result


def top_indices(values: np.ndarray, top_p: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError("need at least two scores")
    count = min(max(1, int(np.ceil(top_p * values.size))), values.size // 2)
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def expanded_indices(values: np.ndarray, top_p: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    count = max(1, min(int(np.ceil(top_p * values.size)), values.size))
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def _normal_score_threshold(
    dataset: str,
    labels: dict[str, str],
    pseudo_paths: dict[str, tuple[str, str]],
    quantile: float,
    per_video: int,
) -> tuple[float, int]:
    samples = []
    for key, label in sorted(labels.items()):
        if not is_normal_label(dataset, label) or key not in pseudo_paths:
            continue
        pseudo_label, path = pseudo_paths[key]
        if pseudo_label != label:
            raise ValueError(f"{key}: pseudo label differs from training CSV")
        scores = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
        if scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError(f"{path}: invalid normal score array")
        samples.append(scores[uniform_indices(scores.size, min(per_video, scores.size))])
    if not samples:
        raise RuntimeError("no usable normal pseudo scores")
    all_scores = np.concatenate(samples)
    return float(np.quantile(all_scores, quantile)), int(all_scores.size)


def _hidden_mean_std(normal_paths: list[str], per_video: int) -> tuple[np.ndarray, np.ndarray, int]:
    count = 0
    mean = m2 = None
    for path in tqdm(normal_paths, desc="normal hidden mean/std", unit="video"):
        hidden, _metadata = load_hidden(path)
        if hidden.ndim != 3 or hidden.shape[0] == 0:
            raise ValueError(f"{path}: expected non-empty [T,L,D], got {hidden.shape}")
        for snippet in hidden[uniform_indices(hidden.shape[0], min(per_video, hidden.shape[0]))]:
            if mean is None:
                mean, m2 = np.zeros_like(snippet, dtype=np.float64), np.zeros_like(snippet, dtype=np.float64)
            elif snippet.shape != mean.shape:
                raise ValueError(f"{path}: hidden shape differs from normal reference")
            count += 1
            delta = snippet - mean
            mean += delta / count
            m2 += delta * (snippet - mean)
    if count < 2 or mean is None or m2 is None:
        raise RuntimeError("need at least two normal hidden snippets")
    return mean.astype(np.float32), np.sqrt(np.maximum(m2 / (count - 1), 1e-12)).astype(np.float32), count


def _normal_novelty_distribution(
    normal_paths: list[str], mean: np.ndarray, std: np.ndarray, per_video: int, sigma_min: float,
) -> np.ndarray:
    values = []
    for path in tqdm(normal_paths, desc="normal novelty calibration", unit="video"):
        hidden, _metadata = load_hidden(path)
        if hidden.ndim != 3 or hidden.shape[1:] != mean.shape:
            raise ValueError(f"{path}: hidden shape differs from normal mean/std")
        chosen = hidden[uniform_indices(hidden.shape[0], min(per_video, hidden.shape[0]))]
        z = (chosen - mean) / (std + sigma_min)
        values.append(np.sqrt(np.mean(np.square(z, dtype=np.float64), axis=(1, 2))).astype(np.float32))
    return np.concatenate(values)


def load_or_build_calibration(
    out_dir: Path,
    dataset: str,
    source_train_csv: str,
    hidden_manifest: str,
    pseudo_csv: str,
    normal_score_quantile: float,
    normal_novelty_quantile: float,
    score_temperature: float,
    novelty_temperature_scale: float,
    seed_top_p: float,
    expand_top_p: float,
    normal_score_snippets_per_video: int,
    normal_hidden_snippets_per_video: int,
    sigma_min: float,
    no_resume: bool,
) -> tuple[NoveltyCalibration, np.ndarray, np.ndarray, dict[str, object]]:
    """Fit or reuse the normal-only calibration persisted in this diagnostic output."""
    cache = out_dir / "normal_novelty_calibration.npz"
    if cache.is_file() and not no_resume:
        with np.load(cache, allow_pickle=False) as artifact:
            required = {"mean", "std", "score_threshold", "novelty_threshold", "novelty_temperature"}
            if not required.issubset(artifact.files):
                raise ValueError(f"{cache}: incomplete calibration; use --clean or --no-resume")
            mean, std = np.asarray(artifact["mean"], dtype=np.float32), np.asarray(artifact["std"], dtype=np.float32)
            calibration = NoveltyCalibration(
                score_threshold=float(artifact["score_threshold"].item()), score_temperature=float(score_temperature),
                novelty_threshold=float(artifact["novelty_threshold"].item()),
                novelty_temperature=float(artifact["novelty_temperature"].item()), seed_top_p=float(seed_top_p),
                expand_top_p=float(expand_top_p), sigma_min=float(sigma_min),
            )
            audit = {"normal_score_samples": int(artifact["normal_score_samples"].item()), "normal_hidden_samples": int(artifact["normal_hidden_samples"].item()), "reused": True}
        return calibration, mean, std, audit

    if not 0.5 < normal_score_quantile < 1.0 or not 0.5 < normal_novelty_quantile < 1.0:
        raise ValueError("normal score/novelty quantiles must be in (0.5, 1.0)")
    if not 0.0 < seed_top_p <= expand_top_p <= 0.5 or score_temperature <= 0.0 or novelty_temperature_scale <= 0.0:
        raise ValueError("invalid reliability parameters")
    labels, hidden_paths, pseudo_paths = labels_by_key(source_train_csv), manifest_map(hidden_manifest), pseudo_score_paths(pseudo_csv)
    normal_paths = [hidden_paths[key] for key, label in labels.items() if is_normal_label(dataset, label) and key in hidden_paths]
    if not normal_paths:
        raise RuntimeError("no normal training hidden artifacts")
    score_threshold, score_samples = _normal_score_threshold(dataset, labels, pseudo_paths, normal_score_quantile, normal_score_snippets_per_video)
    mean, std, hidden_samples = _hidden_mean_std(normal_paths, normal_hidden_snippets_per_video)
    novelty_values = _normal_novelty_distribution(normal_paths, mean, std, normal_hidden_snippets_per_video, sigma_min)
    novelty_threshold = float(np.quantile(novelty_values, normal_novelty_quantile))
    # One temperature unit equals the normal q95-to-q99 tail width.  This
    # avoids hard-coding an architecture-dependent hidden-distance scale.
    novelty_q99 = float(np.quantile(novelty_values, 0.99))
    novelty_temperature = max((novelty_q99 - novelty_threshold) * novelty_temperature_scale, 1e-4)
    calibration = NoveltyCalibration(score_threshold, score_temperature, novelty_threshold, novelty_temperature, seed_top_p, expand_top_p, sigma_min)
    np.savez_compressed(
        cache, mean=mean, std=std, score_threshold=np.asarray(score_threshold), novelty_threshold=np.asarray(novelty_threshold),
        novelty_temperature=np.asarray(novelty_temperature), normal_score_samples=np.asarray(score_samples), normal_hidden_samples=np.asarray(hidden_samples),
        normal_novelty_q99=np.asarray(novelty_q99),
    )
    return calibration, mean, std, {"normal_score_samples": score_samples, "normal_hidden_samples": hidden_samples, "reused": False}


def reliability_map(hidden: np.ndarray, scores: np.ndarray, mean: np.ndarray, std: np.ndarray, calibration: NoveltyCalibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q requires both an anomalous score and a departure from training normality."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[0] == 0 or hidden.shape[1:] != mean.shape or std.shape != mean.shape:
        raise ValueError("hidden sequence and normal statistics have incompatible shapes")
    aligned_scores = resample_scores(scores, hidden.shape[0])
    z = (hidden - mean) / (std + calibration.sigma_min)
    novelty = np.sqrt(np.mean(np.square(z, dtype=np.float64), axis=(1, 2))).astype(np.float32)
    score_confidence = 1.0 / (1.0 + np.exp(-np.clip((aligned_scores - calibration.score_threshold) / calibration.score_temperature, -50.0, 50.0)))
    novelty_confidence = 1.0 / (1.0 + np.exp(-np.clip((novelty - calibration.novelty_threshold) / calibration.novelty_temperature, -50.0, 50.0)))
    return np.clip(score_confidence * novelty_confidence, 0.0, 1.0).astype(np.float32), aligned_scores, novelty


def calibration_dict(calibration: NoveltyCalibration) -> dict[str, float | str]:
    return {
        "seed_top_p": calibration.seed_top_p, "expand_top_p": calibration.expand_top_p,
        "normal_score_threshold": calibration.score_threshold, "score_temperature": calibration.score_temperature,
        "normal_novelty_threshold": calibration.novelty_threshold, "novelty_temperature": calibration.novelty_temperature,
        "sigma_min": calibration.sigma_min,
        "reliability_definition": "sigmoid((classifier_prob1-normal_score_threshold)/score_temperature) * sigmoid((normalised_hidden_RMS-normal_novelty_threshold)/novelty_temperature)",
    }
