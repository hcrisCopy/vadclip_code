#!/usr/bin/env python3
"""Select CLIP neurons with VadCLIP-guided intra-video paired ShiftScore.

Each abnormal training video supplies both ends of its own pseudo-score
ranking: top-p snippets are pseudo-abnormal and bottom-p snippets are
pseudo-normal.  Pure normal videos only estimate the z-score statistics used
when producing the selected-neuron feature.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
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
    frame = pd.read_csv(path)
    missing = {"key", "hidden_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns: {sorted(missing)}")
    return {str(row["key"]): str(row["hidden_path"]) for _, row in frame.iterrows()}


def manifest_token_pool(path: str) -> str:
    frame = pd.read_csv(path)
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame.columns else {"cls"}
    if pools - {"cls", "patch_mean"} or len(pools) != 1:
        raise ValueError(f"{path}: expected exactly one valid token_pool, got {sorted(pools)}")
    return next(iter(pools))


def label_map(path: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for _, row in read_csv(path).iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"{path}: video {key!r} has inconsistent labels")
        labels[key] = label
    return labels


def pseudo_score_map(path: str) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(path)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing pseudo-score columns: {sorted(missing)}")
    return {str(row["key"]): (str(row["label"]), str(row["score_path"])) for _, row in frame.iterrows()}


def collect_normal_stats(hidden_paths: list[str], limit_per_video: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate normal [layer,dim] mean/std with equal-capped video sampling."""
    count = 0
    mean = m2 = None
    for hidden_path in tqdm(hidden_paths, desc="normal z-score statistics", unit="video"):
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
            delta = snippet - mean
            mean += delta / count
            m2 += delta * (snippet - mean)
    if count < 2 or mean is None or m2 is None:
        raise RuntimeError("Need at least two pure-normal hidden snippets for z-score statistics")
    return mean.astype(np.float32), np.sqrt(np.maximum(m2 / (count - 1), 1e-12)).astype(np.float32)


def paired_indices(scores: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) < 2:
        raise ValueError(f"need at least two scores, got {len(scores)}")
    requested = max(1, int(np.ceil(top_p * len(scores))))
    count = min(requested, len(scores) // 2)
    order = np.argsort(scores, kind="mergesort")
    bottom, top = order[:count], order[-count:][::-1]
    if np.intersect1d(top, bottom).size:
        raise RuntimeError("top and bottom pseudo-score sets overlap")
    return top.astype(np.int64), bottom.astype(np.int64)


def global_top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    """Return deterministic flat indices for the global top-k score entries."""
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if topk <= 0 or topk > flat_scores.size:
        raise ValueError(f"topk={topk} must be in [1, {flat_scores.size}]")
    return np.argsort(-flat_scores, kind="mergesort")[:topk]


def bootstrap_stability_frequency(
    delta_array: np.ndarray,
    topk: int,
    repeats: int,
    fraction: float,
    seed: int,
    sigma_min: float,
) -> tuple[np.ndarray, int]:
    """Estimate global-top-k inclusion frequency by video-level subsampling.

    Every repeat samples whole abnormal videos without replacement.  This keeps
    snippets from one video together, so the paired-effect unit remains the
    same video-level delta used by the main ShiftScore calculation.
    """
    if repeats <= 0:
        return np.ones(delta_array.shape[1:], dtype=np.float32), int(delta_array.shape[0])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("bootstrap fraction must be in (0, 1]")
    if delta_array.shape[0] < 2:
        raise ValueError("at least two abnormal videos are required for stability selection")

    sampled_videos = max(2, int(np.ceil(fraction * delta_array.shape[0])))
    sampled_videos = min(sampled_videos, int(delta_array.shape[0]))
    rng = np.random.default_rng(seed)
    hits = np.zeros(delta_array.shape[1:], dtype=np.int32)
    for _ in tqdm(range(repeats), desc="global selection stability", unit="repeat"):
        indices = rng.choice(delta_array.shape[0], size=sampled_videos, replace=False)
        sampled = delta_array[indices]
        mean_delta = sampled.mean(axis=0)
        std_delta = sampled.std(axis=0, ddof=1)
        sampled_scores = np.abs(mean_delta) / (std_delta + sigma_min)
        hits.reshape(-1)[global_top_indices(sampled_scores, topk)] += 1
    return (hits.astype(np.float32) / float(repeats)), sampled_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Select CLIP neurons by intra-video paired ShiftScore.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--topk-per-layer", type=int, default=None)
    selector.add_argument("--topk-global", type=int, default=None)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument(
        "--bootstrap-repeats", type=int, default=0,
        help="With global selection, repeat video-level subsampling and rank by ShiftScore × inclusion frequency.",
    )
    parser.add_argument(
        "--bootstrap-fraction", type=float, default=0.80,
        help="Fraction of valid abnormal videos sampled without replacement per stability repeat.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.topk_per_layer is None and args.topk_global is None:
        args.topk_per_layer = 64

    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")
    if args.topk_per_layer is not None and args.topk_per_layer <= 0:
        parser.error("--topk-per-layer must be positive")
    if args.topk_global is not None and args.topk_global <= 0:
        parser.error("--topk-global must be positive")
    if args.bootstrap_repeats < 0:
        parser.error("--bootstrap-repeats must be non-negative")
    if not 0.0 < args.bootstrap_fraction <= 1.0:
        parser.error("--bootstrap-fraction must be in (0, 1]")
    if args.bootstrap_repeats and args.topk_global is None:
        parser.error("--bootstrap-repeats requires --topk-global so the final selection remains global")
    if args.normal_stat_snippets_per_video <= 0 or args.sigma_min <= 0:
        parser.error("normal-stat-snippets-per-video and sigma-min must be positive")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    output_json = out_dir / "selected_neurons.json"
    if output_json.exists() and not args.no_resume:
        print(f"reuse completed selection: {output_json}", flush=True)
        return

    hidden_by_key = manifest_map(args.hidden_manifest)
    token_pool = manifest_token_pool(args.hidden_manifest)
    labels_by_key = label_map(args.source_train_csv)
    scores_by_key = pseudo_score_map(args.pseudo_csv)
    normal_keys, abnormal_keys, skipped = [], [], []
    for key, label in sorted(labels_by_key.items()):
        if is_normal_label(args.dataset, label):
            if key in hidden_by_key:
                normal_keys.append(key)
            else:
                skipped.append([key, label, "normal", "missing_hidden"])
        elif key not in hidden_by_key:
            skipped.append([key, label, "abnormal", "missing_hidden"])
        elif key not in scores_by_key:
            skipped.append([key, label, "abnormal", "missing_pseudo_score"])
        else:
            abnormal_keys.append(key)
    if not normal_keys or not abnormal_keys:
        raise RuntimeError(f"matched normal={len(normal_keys)}, abnormal={len(abnormal_keys)}; both are required")
    print(f"matched normal_for_stats={len(normal_keys)}, abnormal_for_pairs={len(abnormal_keys)}", flush=True)

    normal_mean, normal_std = collect_normal_stats(
        [hidden_by_key[key] for key in normal_keys], args.normal_stat_snippets_per_video
    )
    deltas, pair_rows = [], []
    for key in tqdm(abnormal_keys, desc="video-internal paired deltas", unit="video"):
        label = labels_by_key[key]
        hidden, _metadata = load_hidden(hidden_by_key[key])
        if hidden.ndim != 3 or hidden.shape[0] == 0:
            skipped.append([key, label, "abnormal", f"invalid_hidden_shape_{tuple(hidden.shape)}"])
            continue
        if hidden.shape[1:] != normal_mean.shape:
            raise ValueError(f"{key}: hidden [L,D]={hidden.shape[1:]} differs from normal stats {normal_mean.shape}")
        score_label, score_path = scores_by_key[key]
        if score_label != label:
            raise ValueError(f"{key}: pseudo-score label {score_label!r} differs from source label {label!r}")
        raw_scores = np.load(score_path)
        aligned_scores = resample_scores(raw_scores, hidden.shape[0])
        try:
            positive, negative = paired_indices(aligned_scores, args.top_p)
        except ValueError as error:
            skipped.append([key, label, "abnormal", str(error)])
            continue
        z_hidden = (hidden - normal_mean) / (normal_std + args.sigma_min)
        delta = z_hidden[positive].mean(axis=0) - z_hidden[negative].mean(axis=0)
        if not np.isfinite(delta).all():
            raise RuntimeError(f"{key}: non-finite paired delta")
        deltas.append(delta.astype(np.float32))
        pair_rows.append([
            key, label, int(hidden.shape[0]), int(len(raw_scores)), int(len(positive)),
            float(aligned_scores[positive].mean()), float(aligned_scores[negative].mean()),
        ])
    if len(deltas) < 2:
        raise RuntimeError(f"only {len(deltas)} valid abnormal paired deltas; need at least two")

    delta_array = np.stack(deltas, axis=0)
    mean_delta = delta_array.mean(axis=0)
    std_delta = delta_array.std(axis=0, ddof=1)
    shift_scores = np.abs(mean_delta) / (std_delta + args.sigma_min)
    if not np.isfinite(shift_scores).all():
        raise RuntimeError("non-finite ShiftScore")

    if args.topk_global is not None:
        stability_frequency, sampled_videos = bootstrap_stability_frequency(
            delta_array,
            args.topk_global,
            args.bootstrap_repeats,
            args.bootstrap_fraction,
            args.bootstrap_seed,
            args.sigma_min,
        )
        selection_scores = shift_scores * stability_frequency
    else:
        stability_frequency = np.ones_like(shift_scores, dtype=np.float32)
        selection_scores = shift_scores
        sampled_videos = 0

    selected = []
    if args.topk_global is not None:
        flat = global_top_indices(selection_scores, args.topk_global)
        for layer in range(shift_scores.shape[0]):
            dims = (flat[flat // shift_scores.shape[1] == layer] % shift_scores.shape[1]).astype(np.int64)
            if len(dims):
                selected.append({
                    "layer_index": int(layer), "dims": dims.tolist(),
                    "scores": selection_scores[layer, dims].astype(float).tolist(),
                    "shift_scores": shift_scores[layer, dims].astype(float).tolist(),
                    "stability_frequency": stability_frequency[layer, dims].astype(float).tolist(),
                    "mean_deltas": mean_delta[layer, dims].astype(float).tolist(),
                    "std_deltas": std_delta[layer, dims].astype(float).tolist(),
                    "directions": np.sign(mean_delta[layer, dims]).astype(int).tolist(),
                })
        visual_width, mode = int(args.topk_global), "global"
    else:
        if args.topk_per_layer > shift_scores.shape[1]:
            raise ValueError("--topk-per-layer exceeds hidden dimension")
        for layer in range(shift_scores.shape[0]):
            dims = np.argsort(shift_scores[layer], kind="mergesort")[-args.topk_per_layer:][::-1]
            selected.append({
                "layer_index": int(layer), "dims": dims.astype(int).tolist(),
                "scores": shift_scores[layer, dims].astype(float).tolist(),
                "mean_deltas": mean_delta[layer, dims].astype(float).tolist(),
                "std_deltas": std_delta[layer, dims].astype(float).tolist(),
                "directions": np.sign(mean_delta[layer, dims]).astype(int).tolist(),
            })
        visual_width, mode = int(shift_scores.shape[0] * args.topk_per_layer), "per_layer"

    normal_mean_path, normal_std_path = out_dir / "normal_mean.npy", out_dir / "normal_std.npy"
    np.save(normal_mean_path, normal_mean)
    np.save(normal_std_path, normal_std)
    np.save(out_dir / "mean_delta.npy", mean_delta.astype(np.float32))
    np.save(out_dir / "std_delta.npy", std_delta.astype(np.float32))
    np.save(out_dir / "shift_scores.npy", shift_scores.astype(np.float32))
    np.save(out_dir / "stability_frequency.npy", stability_frequency.astype(np.float32))
    np.save(out_dir / "selection_scores.npy", selection_scores.astype(np.float32))
    selection_method = (
        "global_shift_score_times_bootstrap_frequency"
        if args.topk_global is not None and args.bootstrap_repeats
        else "shift_score"
    )
    save_json(output_json, {
        "method": "vadclip_intravideo_paired_shift_v1",
        "dataset": args.dataset,
        "description": "Per abnormal video: z-scored top pseudo-score hidden mean minus bottom pseudo-score hidden mean; rank by absolute cross-video paired effect size.",
        "top_p": float(args.top_p), "pairing": "within_abnormal_video_equal_top_bottom_count",
        "normal_videos_used_for_shift_pairs": False, "normal_videos_used_for_zscore_only": True,
        "normal_stat_snippets_per_video": int(args.normal_stat_snippets_per_video), "sigma_min": float(args.sigma_min),
        "num_normal_videos_for_stats": len(normal_keys), "num_abnormal_videos_with_pairs": len(deltas),
        "num_layers": int(shift_scores.shape[0]), "hidden_dim": int(shift_scores.shape[1]),
        "token_pool": token_pool, "selection_mode": mode, "topk_per_layer": args.topk_per_layer,
        "topk_global": args.topk_global, "visual_width": visual_width,
        "selection_score": selection_method,
        "bootstrap_stability": {
            "enabled": bool(args.bootstrap_repeats),
            "repeats": int(args.bootstrap_repeats),
            "fraction": float(args.bootstrap_fraction),
            "seed": int(args.bootstrap_seed),
            "sampling_unit": "abnormal_video_delta_without_replacement",
            "sampled_videos_per_repeat": int(sampled_videos),
            "final_rank": "raw_shift_score_times_global_topk_inclusion_frequency",
        },
        "normal_mean_path": str(normal_mean_path), "normal_std_path": str(normal_std_path),
        "mean_delta_path": str(out_dir / "mean_delta.npy"), "std_delta_path": str(out_dir / "std_delta.npy"),
        "shift_scores_path": str(out_dir / "shift_scores.npy"),
        "stability_frequency_path": str(out_dir / "stability_frequency.npy"),
        "selection_scores_path": str(out_dir / "selection_scores.npy"),
        "source_train_csv": args.source_train_csv,
        "hidden_manifest": args.hidden_manifest, "pseudo_csv": args.pseudo_csv, "selected": selected,
    })
    write_csv(out_dir / "video_pairs.csv", [
        "key", "label", "hidden_snippets", "pseudo_score_len_before_alignment", "paired_count",
        "positive_score_mean", "negative_score_mean",
    ], pair_rows)
    write_csv(out_dir / "skipped_videos.csv", ["key", "label", "role", "reason"], skipped)
    if args.topk_global is not None and args.bootstrap_repeats:
        print(
            f"stability global selection: repeats={args.bootstrap_repeats}, "
            f"fraction={args.bootstrap_fraction:.3f}, sampled_videos={sampled_videos}",
            flush=True,
        )
    print(f"wrote {output_json}: {visual_width} selected dimensions", flush=True)


if __name__ == "__main__":
    main()
