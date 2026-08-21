#!/usr/bin/env python3
"""Select global CLIP neurons from pseudo-top abnormal snippets versus pure normal videos.

The existing global-768 selector subtracts bottom-score snippets inside an
abnormal video.  This experiment instead uses the z-scored pure-normal
training distribution as the negative reference.  In z-score coordinates the
normal reference is exactly zero, so each abnormal video contributes its
top-p pseudo-positive mean minus that pure-normal reference.

No test list, frame annotation, optimiser, or model checkpoint is read here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def add_injection_source() -> None:
    """Use repository-local utilities only; no DSANet code is imported."""
    source = str(Path(__file__).resolve().parents[1] / "vadclip_neuron_injection")
    if source not in sys.path:
        sys.path.insert(0, source)


add_injection_source()
from common import (  # noqa: E402
    base_key,
    clean_dir,
    ensure_dir,
    is_normal_label,
    load_hidden,
    read_csv,
    resample_scores,
    save_json,
    uniform_indices,
    write_csv,
)


def manifest_map(path: str) -> dict[str, str]:
    """Read the shared CLIP-hidden manifest with duplicate-key protection."""
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns: {sorted(missing)}")
    result: dict[str, str] = {}
    for _, row in frame.iterrows():
        key, hidden_path = str(row["key"]), str(row["hidden_path"])
        if key in result and result[key] != hidden_path:
            raise ValueError(f"{path} contains duplicate key {key!r} with different hidden paths")
        result[key] = hidden_path
    return result


def manifest_token_pool(path: str) -> str:
    """Require one common hidden-token pooling contract, as concat building does."""
    frame = pd.read_csv(path)
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame.columns else {"cls"}
    if pools - {"cls", "patch_mean"} or len(pools) != 1:
        raise ValueError(f"{path}: expected exactly one valid token_pool, got {sorted(pools)}")
    return next(iter(pools))


def source_labels(path: str) -> dict[str, str]:
    """Map every video key in the source training CSV to one stable video label."""
    labels: dict[str, str] = {}
    for _, row in read_csv(path).iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"{path}: video {key!r} has inconsistent labels")
        labels[key] = label
    return labels


def pseudo_score_map(path: str) -> dict[str, tuple[str, str]]:
    """Read frozen-baseline score files emitted by score_vadclip_pseudo.py."""
    frame = pd.read_csv(path)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing pseudo-score columns: {sorted(missing)}")
    result: dict[str, tuple[str, str]] = {}
    for _, row in frame.iterrows():
        key, value = str(row["key"]), (str(row["label"]), str(row["score_path"]))
        if key in result:
            raise ValueError(f"{path} contains duplicate pseudo-score key {key!r}")
        result[key] = value
    return result


def collect_normal_stats(hidden_paths: list[str], limit_per_video: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Estimate per-neuron normal mean/std with equal-capped video sampling."""
    count = 0
    mean = m2 = None
    for hidden_path in tqdm(hidden_paths, desc="pure-normal z-score statistics", unit="video"):
        hidden, _metadata = load_hidden(hidden_path)
        if hidden.ndim != 3 or hidden.shape[0] == 0:
            raise ValueError(f"{hidden_path}: expected non-empty [T,L,D], got {hidden.shape}")
        for snippet in hidden[uniform_indices(hidden.shape[0], min(limit_per_video, hidden.shape[0]))]:
            if mean is None:
                mean = np.zeros_like(snippet, dtype=np.float64)
                m2 = np.zeros_like(snippet, dtype=np.float64)
            elif snippet.shape != mean.shape:
                raise ValueError(f"{hidden_path}: layer/dimension shape differs from prior normal videos")
            count += 1
            difference = snippet - mean
            mean += difference / count
            m2 += difference * (snippet - mean)
    if count < 2 or mean is None or m2 is None:
        raise RuntimeError("need at least two pure-normal hidden snippets for normal statistics")
    std = np.sqrt(np.maximum(m2 / (count - 1), 1e-12))
    return mean.astype(np.float32), std.astype(np.float32), count


def load_or_build_normal_stats(
    output_dir: Path,
    normal_paths: list[str],
    limit_per_video: int,
    no_resume: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Reuse completed normal statistics, otherwise build and persist them."""
    cache_path = output_dir / "normal_stats.npz"
    mean_path, std_path = output_dir / "normal_mean.npy", output_dir / "normal_std.npy"
    if cache_path.is_file() and not no_resume:
        with np.load(cache_path, allow_pickle=False) as artifact:
            required = {"mean", "std", "snippet_count"}
            if not required.issubset(artifact.files):
                raise ValueError(f"{cache_path}: incomplete normal-statistics artifact; use --no-resume or --clean")
            mean = np.asarray(artifact["mean"], dtype=np.float32)
            std = np.asarray(artifact["std"], dtype=np.float32)
            count = int(artifact["snippet_count"].item())
        if mean.ndim != 2 or std.shape != mean.shape or count < 2:
            raise ValueError(f"{cache_path}: invalid normal-statistics shape or count; use --no-resume or --clean")
        if not mean_path.is_file():
            np.save(mean_path, mean)
        if not std_path.is_file():
            np.save(std_path, std)
        print(f"reuse pure-normal statistics: {cache_path}", flush=True)
        return mean, std, count

    mean, std, count = collect_normal_stats(normal_paths, limit_per_video)
    np.savez_compressed(cache_path, mean=mean, std=std, snippet_count=np.asarray(count, dtype=np.int64))
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"wrote pure-normal statistics from {count} snippets", flush=True)
    return mean, std, count


def top_indices(scores: np.ndarray, top_p: float) -> np.ndarray:
    """Use the existing global-768 top-count rule while intentionally omitting bottom samples."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError(f"need at least two pseudo scores, got {values.size}")
    requested = max(1, int(np.ceil(float(top_p) * values.size)))
    count = min(requested, values.size // 2)
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def delta_artifact_path(delta_dir: Path, key: str) -> Path:
    """One saved abnormal-video delta enables safe resume after interruption."""
    return delta_dir / f"{key}.npz"


def load_delta(path: Path, key: str, label: str, expected_shape: tuple[int, int]) -> tuple[np.ndarray, list[object]]:
    """Validate a completed per-video contribution before reusing it."""
    with np.load(path, allow_pickle=False) as artifact:
        required = {"delta", "key", "label", "hidden_length", "score_length", "top_count", "top_score_mean"}
        if not required.issubset(artifact.files):
            raise ValueError(f"{path}: incomplete delta artifact; use --no-resume or --clean")
        if str(artifact["key"].item()) != key or str(artifact["label"].item()) != label:
            raise ValueError(f"{path}: key/label differs from the current source CSV; use --clean")
        delta = np.asarray(artifact["delta"], dtype=np.float32)
        row = [
            key, label, int(artifact["hidden_length"].item()), int(artifact["score_length"].item()),
            int(artifact["top_count"].item()), float(artifact["top_score_mean"].item()), "reused",
        ]
    if delta.shape != expected_shape or not np.isfinite(delta).all():
        raise ValueError(f"{path}: invalid delta shape/value; use --no-resume or --clean")
    return delta, row


def global_top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    """Return deterministic global top-k layer/dimension positions."""
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if topk <= 0 or topk > flat.size:
        raise ValueError(f"topk={topk} must be in [1, {flat.size}]")
    return np.argsort(-flat, kind="mergesort")[:topk]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select global neurons from abnormal pseudo-top snippets versus pure normal videos.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--topk-global", type=int, default=768)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")
    if args.topk_global <= 0 or args.normal_stat_snippets_per_video <= 0 or args.sigma_min <= 0.0:
        parser.error("topk-global, normal-stat-snippets-per-video, and sigma-min must be positive")
    for path in (args.source_train_csv, args.hidden_manifest, args.pseudo_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing top-vs-normal selection input: {path}")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    output_json = out_dir / "selected_neurons.json"
    if output_json.is_file() and not args.no_resume:
        print(f"reuse completed top-vs-normal global selection: {output_json}", flush=True)
        return

    hidden_by_key = manifest_map(args.hidden_manifest)
    token_pool = manifest_token_pool(args.hidden_manifest)
    labels_by_key = source_labels(args.source_train_csv)
    scores_by_key = pseudo_score_map(args.pseudo_csv)
    normal_keys, abnormal_keys = [], []
    for key, label in labels_by_key.items():
        if key not in hidden_by_key:
            raise FileNotFoundError(f"{args.hidden_manifest}: missing hidden artifact for source video {key!r}")
        if is_normal_label(args.dataset, label):
            normal_keys.append(key)
        else:
            if key not in scores_by_key:
                raise KeyError(f"{args.pseudo_csv}: missing abnormal video {key!r}")
            pseudo_label, _score_path = scores_by_key[key]
            if pseudo_label != label:
                raise ValueError(f"{key}: pseudo label {pseudo_label!r} differs from source label {label!r}")
            abnormal_keys.append(key)
    if not normal_keys or not abnormal_keys:
        raise RuntimeError(f"matched normal={len(normal_keys)}, abnormal={len(abnormal_keys)}; both are required")
    print(f"matched pure-normal reference videos={len(normal_keys)}, pseudo-positive abnormal videos={len(abnormal_keys)}", flush=True)

    normal_mean, normal_std, normal_count = load_or_build_normal_stats(
        out_dir,
        [hidden_by_key[key] for key in normal_keys],
        args.normal_stat_snippets_per_video,
        args.no_resume,
    )
    delta_dir = ensure_dir(out_dir / "per_video_deltas")
    deltas, rows = [], []
    expected_shape = tuple(normal_mean.shape)
    for key in tqdm(abnormal_keys, desc="top-vs-normal video deltas", unit="video"):
        label = labels_by_key[key]
        artifact_path = delta_artifact_path(delta_dir, key)
        if artifact_path.is_file() and not args.no_resume:
            delta, row = load_delta(artifact_path, key, label, expected_shape)
        else:
            hidden, _metadata = load_hidden(hidden_by_key[key])
            if hidden.ndim != 3 or hidden.shape[0] == 0 or tuple(hidden.shape[1:]) != expected_shape:
                raise ValueError(f"{key}: hidden shape {hidden.shape} does not match normal-stat shape {expected_shape}")
            _pseudo_label, score_path = scores_by_key[key]
            raw_scores = np.asarray(np.load(score_path, allow_pickle=False), dtype=np.float32).reshape(-1)
            if raw_scores.size == 0 or not np.isfinite(raw_scores).all():
                raise ValueError(f"{score_path}: expected finite non-empty pseudo scores")
            aligned_scores = resample_scores(raw_scores, hidden.shape[0])
            positive = top_indices(aligned_scores, args.top_p)
            z_hidden = (hidden - normal_mean) / (normal_std + args.sigma_min)
            # The pure-normal bank has mean zero in this z-score coordinate.
            delta = z_hidden[positive].mean(axis=0).astype(np.float32)
            if not np.isfinite(delta).all():
                raise RuntimeError(f"{key}: non-finite top-vs-normal delta")
            np.savez_compressed(
                artifact_path,
                delta=delta,
                key=np.asarray(key),
                label=np.asarray(label),
                hidden_length=np.asarray(hidden.shape[0], dtype=np.int64),
                score_length=np.asarray(raw_scores.size, dtype=np.int64),
                top_count=np.asarray(positive.size, dtype=np.int64),
                top_score_mean=np.asarray(aligned_scores[positive].mean(), dtype=np.float32),
            )
            row = [key, label, int(hidden.shape[0]), int(raw_scores.size), int(positive.size), float(aligned_scores[positive].mean()), "computed"]
        deltas.append(delta)
        rows.append(row)
    if len(deltas) < 2:
        raise RuntimeError(f"only {len(deltas)} usable abnormal-video deltas; need at least two")

    delta_array = np.stack(deltas, axis=0)
    mean_delta = delta_array.mean(axis=0)
    std_delta = delta_array.std(axis=0, ddof=1)
    selection_scores = np.abs(mean_delta) / (std_delta + args.sigma_min)
    if not np.isfinite(selection_scores).all():
        raise RuntimeError("non-finite top-vs-normal ShiftScore")
    flat = global_top_indices(selection_scores, args.topk_global)
    selected = []
    hidden_dim = selection_scores.shape[1]
    for layer in range(selection_scores.shape[0]):
        dims = (flat[flat // hidden_dim == layer] % hidden_dim).astype(np.int64)
        if dims.size:
            selected.append({
                "layer_index": int(layer),
                "dims": dims.tolist(),
                "scores": selection_scores[layer, dims].astype(float).tolist(),
                "mean_deltas": mean_delta[layer, dims].astype(float).tolist(),
                "std_deltas": std_delta[layer, dims].astype(float).tolist(),
                "directions": np.sign(mean_delta[layer, dims]).astype(int).tolist(),
            })
    selected_width = sum(len(item["dims"]) for item in selected)
    if selected_width != args.topk_global:
        raise RuntimeError(f"selected width={selected_width}, expected {args.topk_global}")

    mean_path, std_path = out_dir / "mean_delta.npy", out_dir / "std_delta.npy"
    score_path = out_dir / "selection_scores.npy"
    np.save(mean_path, mean_delta.astype(np.float32))
    np.save(std_path, std_delta.astype(np.float32))
    np.save(score_path, selection_scores.astype(np.float32))
    write_csv(
        out_dir / "per_video_top_rows.csv",
        ["key", "label", "hidden_length", "raw_score_length", "top_count", "top_score_mean", "status"],
        rows,
    )
    save_json(output_json, {
        "method": "vadclip_top_vs_pure_normal_global_shift_v1",
        "dataset": args.dataset,
        "description": "Per abnormal video: z-scored hidden mean of top pseudo-score snippets minus the pure-normal z-score reference (zero); globally rank by absolute cross-video effect size.",
        "positive_definition": "top pseudo-score snippets within each abnormal training video",
        "negative_definition": "pure normal training-video reference distribution, represented by zero after normal z-scoring",
        "top_p": float(args.top_p),
        "normal_videos_used_for_reference": True,
        "abnormal_bottom_snippets_used_as_negatives": False,
        "frame_labels_used": False,
        "test_data_used": False,
        "normal_stat_snippets_per_video": int(args.normal_stat_snippets_per_video),
        "sigma_min": float(args.sigma_min),
        "num_normal_videos_for_reference": len(normal_keys),
        "num_normal_snippets_for_reference": int(normal_count),
        "num_abnormal_videos_with_top_deltas": len(deltas),
        "num_layers": int(selection_scores.shape[0]),
        "hidden_dim": int(selection_scores.shape[1]),
        "token_pool": token_pool,
        "selection_mode": "global",
        "topk_global": int(args.topk_global),
        "visual_width": int(args.topk_global),
        "selection_score": "absolute_mean_top_vs_pure_normal_z_delta_over_cross_video_std",
        "normal_mean_path": str(out_dir / "normal_mean.npy"),
        "normal_std_path": str(out_dir / "normal_std.npy"),
        "mean_delta_path": str(mean_path),
        "std_delta_path": str(std_path),
        "selection_scores_path": str(score_path),
        "per_video_delta_dir": str(delta_dir),
        "selected": selected,
    })
    print(
        f"wrote {output_json}: global top-{args.topk_global}; "
        f"positive=abnormal top-{args.top_p:.3f}, negative=pure-normal reference",
        flush=True,
    )


if __name__ == "__main__":
    main()
