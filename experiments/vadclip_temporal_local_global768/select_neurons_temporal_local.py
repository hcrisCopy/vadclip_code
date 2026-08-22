#!/usr/bin/env python3
"""Select global CLIP neurons with temporal-local and pure-normal agreement.

For every abnormal training video, frozen-baseline top-p snippets form a
pseudo-positive set.  Its selected neurons must satisfy two independent
conditions:

1. distinguish the pseudo-positive snippets from temporally distant,
   lower-score snippets in the *same* video; and
2. distinguish the pseudo-positive snippets from the pure-normal training
   distribution.

Only cached hidden states and frozen-baseline pseudo scores are used.  Test
data, frame annotations, and model training are deliberately outside this
selection script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def add_injection_source() -> None:
    """Expose only repository-local feature and file utilities."""
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
    """Read the hidden manifest and reject ambiguous video keys."""
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
    """Require the same hidden-token pooling contract across all rows."""
    frame = pd.read_csv(path)
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame.columns else {"cls"}
    if pools - {"cls", "patch_mean"} or len(pools) != 1:
        raise ValueError(f"{path}: expected exactly one valid token_pool, got {sorted(pools)}")
    return next(iter(pools))


def source_labels(path: str) -> dict[str, str]:
    """Map each training video key to its one stable weak video label."""
    labels: dict[str, str] = {}
    for _, row in read_csv(path).iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"{path}: video {key!r} has inconsistent labels")
        labels[key] = label
    return labels


def pseudo_score_map(path: str) -> dict[str, tuple[str, str]]:
    """Read score paths emitted by the frozen VadCLIP pseudo-score stage."""
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
    """Estimate normal per-neuron mean/std with equal-capped video sampling."""
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
    """Persist normal statistics so an interrupted selector can resume safely."""
    cache_path = output_dir / "normal_stats.npz"
    mean_path, std_path = output_dir / "normal_mean.npy", output_dir / "normal_std.npy"
    if cache_path.is_file() and not no_resume:
        with np.load(cache_path, allow_pickle=False) as artifact:
            required = {"mean", "std", "snippet_count"}
            if not required.issubset(artifact.files):
                raise ValueError(f"{cache_path}: incomplete normal statistics; use --no-resume or --clean")
            mean = np.asarray(artifact["mean"], dtype=np.float32)
            std = np.asarray(artifact["std"], dtype=np.float32)
            count = int(artifact["snippet_count"].item())
        if mean.ndim != 2 or std.shape != mean.shape or count < 2:
            raise ValueError(f"{cache_path}: invalid normal-statistics shape/count; use --no-resume or --clean")
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
    """Match the established deterministic per-video top-p count rule."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError(f"need at least two pseudo scores, got {values.size}")
    requested = max(1, int(np.ceil(float(top_p) * values.size)))
    count = min(requested, values.size // 2)
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def score_ranks(scores: np.ndarray) -> np.ndarray:
    """Return deterministic within-video ranks in [0, 1], retaining tied order."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError("at least two pseudo scores are required for temporal ranks")
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[np.argsort(values, kind="mergesort")] = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    return ranks


def temporal_support_weights(scores: np.ndarray, positive: np.ndarray, context_radius: int) -> np.ndarray:
    """Downweight isolated pseudo-score spikes without discarding top-p snippets."""
    ranks = score_ranks(scores)
    support = np.empty(positive.size, dtype=np.float32)
    for output_index, snippet_index in enumerate(positive.tolist()):
        left = max(0, snippet_index - context_radius)
        right = min(ranks.size, snippet_index + context_radius + 1)
        support[output_index] = ranks[left:right].mean(dtype=np.float32)
    # Every selected top-p snippet retains at least half of its original vote.
    return (0.5 + 0.5 * support).astype(np.float32)


def background_indices(
    scores: np.ndarray,
    positive: np.ndarray,
    context_radius: int,
    background_p: float,
) -> np.ndarray:
    """Choose low-score, temporally distant same-video background snippets."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    blocked = np.zeros(values.size, dtype=bool)
    for snippet_index in positive.tolist():
        left = max(0, snippet_index - context_radius)
        right = min(values.size, snippet_index + context_radius + 1)
        blocked[left:right] = True
    eligible = np.flatnonzero(~blocked)
    if eligible.size == 0:
        raise ValueError("positive temporal neighborhoods cover the whole video; reduce --context-radius or --top-p")
    requested = max(1, int(np.ceil(float(background_p) * eligible.size)))
    count = min(requested, eligible.size)
    local_order = np.argsort(values[eligible], kind="mergesort")[:count]
    return eligible[local_order].astype(np.int64)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average a non-empty [T,L,D] set over time using positive scalar weights."""
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError(f"expected non-empty [T,L,D] values, got {values.shape}")
    sample_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if sample_weights.shape[0] != values.shape[0] or not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0.0):
        raise ValueError("weights must be finite, strictly positive, and match the temporal dimension")
    return np.average(values, axis=0, weights=sample_weights).astype(np.float32)


def contribution_artifact_path(contribution_dir: Path, key: str) -> Path:
    """One artifact per abnormal video provides interruption-safe progress."""
    return contribution_dir / f"{key}.npz"


def load_contribution(
    path: Path,
    key: str,
    label: str,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list[object]]:
    """Validate one completed video contribution before it is reused."""
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "event_delta", "normal_delta", "key", "label", "hidden_length", "score_length",
            "positive_count", "background_count", "top_score_mean", "background_score_mean",
            "positive_weight_mean", "temporal_support_mean",
        }
        if not required.issubset(artifact.files):
            raise ValueError(f"{path}: incomplete contribution artifact; use --no-resume or --clean")
        if str(artifact["key"].item()) != key or str(artifact["label"].item()) != label:
            raise ValueError(f"{path}: key/label differs from the current source CSV; use --clean")
        event_delta = np.asarray(artifact["event_delta"], dtype=np.float32)
        normal_delta = np.asarray(artifact["normal_delta"], dtype=np.float32)
        row = [
            key, label, int(artifact["hidden_length"].item()), int(artifact["score_length"].item()),
            int(artifact["positive_count"].item()), int(artifact["background_count"].item()),
            float(artifact["top_score_mean"].item()), float(artifact["background_score_mean"].item()),
            float(artifact["positive_weight_mean"].item()), float(artifact["temporal_support_mean"].item()), "reused",
        ]
    if event_delta.shape != expected_shape or normal_delta.shape != expected_shape:
        raise ValueError(f"{path}: contribution shape differs from normal statistics; use --clean")
    if not np.isfinite(event_delta).all() or not np.isfinite(normal_delta).all():
        raise ValueError(f"{path}: non-finite contribution; use --clean")
    return event_delta, normal_delta, row


def global_top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    """Return deterministic global top-k layer/dimension positions."""
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if topk <= 0 or topk > flat.size:
        raise ValueError(f"topk={topk} must be in [1, {flat.size}]")
    return np.argsort(-flat, kind="mergesort")[:topk]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select global neurons using pseudo-positive temporal-local contrast and pure-normal agreement."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--background-p", type=float, default=0.50)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--topk-global", type=int, default=768)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")
    if not 0.0 < args.background_p <= 1.0:
        parser.error("--background-p must be in (0, 1]")
    if args.context_radius < 0:
        parser.error("--context-radius must be non-negative")
    if args.topk_global <= 0 or args.normal_stat_snippets_per_video <= 0 or args.sigma_min <= 0.0:
        parser.error("topk-global, normal-stat-snippets-per-video, and sigma-min must be positive")
    for path in (args.source_train_csv, args.hidden_manifest, args.pseudo_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing temporal-local selection input: {path}")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    output_json = out_dir / "selected_neurons.json"
    if output_json.is_file() and not args.no_resume:
        print(f"reuse completed temporal-local global selection: {output_json}", flush=True)
        return

    hidden_by_key = manifest_map(args.hidden_manifest)
    token_pool = manifest_token_pool(args.hidden_manifest)
    labels_by_key = source_labels(args.source_train_csv)
    scores_by_key = pseudo_score_map(args.pseudo_csv)
    normal_keys, abnormal_keys, skipped = [], [], []
    for key, label in sorted(labels_by_key.items()):
        role = "normal" if is_normal_label(args.dataset, label) else "abnormal"
        if key not in hidden_by_key:
            skipped.append([key, label, role, "missing_hidden"])
            continue
        if role == "normal":
            normal_keys.append(key)
            continue
        if key not in scores_by_key:
            skipped.append([key, label, role, "missing_pseudo_score"])
            continue
        pseudo_label, _score_path = scores_by_key[key]
        if pseudo_label != label:
            raise ValueError(f"{key}: pseudo label {pseudo_label!r} differs from source label {label!r}")
        abnormal_keys.append(key)
    if not normal_keys or not abnormal_keys:
        raise RuntimeError(f"matched normal={len(normal_keys)}, abnormal={len(abnormal_keys)}; both are required")
    print(
        f"matched pure-normal reference videos={len(normal_keys)}, "
        f"pseudo-positive abnormal videos={len(abnormal_keys)}",
        flush=True,
    )

    normal_mean, normal_std, normal_count = load_or_build_normal_stats(
        out_dir,
        [hidden_by_key[key] for key in normal_keys],
        args.normal_stat_snippets_per_video,
        args.no_resume,
    )
    expected_shape = tuple(normal_mean.shape)
    contribution_dir = ensure_dir(out_dir / "per_video_contributions")
    event_deltas, normal_deltas, rows = [], [], []
    for key in tqdm(abnormal_keys, desc="temporal-local video contributions", unit="video"):
        label = labels_by_key[key]
        artifact_path = contribution_artifact_path(contribution_dir, key)
        if artifact_path.is_file() and not args.no_resume:
            event_delta, normal_delta, row = load_contribution(artifact_path, key, label, expected_shape)
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
            background = background_indices(aligned_scores, positive, args.context_radius, args.background_p)
            positive_weights = temporal_support_weights(aligned_scores, positive, args.context_radius)
            z_hidden = (hidden - normal_mean) / (normal_std + args.sigma_min)
            positive_mean = weighted_mean(z_hidden[positive], positive_weights)
            background_mean = z_hidden[background].mean(axis=0, dtype=np.float32)
            event_delta = (positive_mean - background_mean).astype(np.float32)
            # The pure-normal reference has mean zero after the same z-scoring.
            normal_delta = positive_mean.astype(np.float32)
            if not np.isfinite(event_delta).all() or not np.isfinite(normal_delta).all():
                raise RuntimeError(f"{key}: non-finite temporal-local contribution")
            ranks = score_ranks(aligned_scores)
            support_mean = float((2.0 * positive_weights - 1.0).mean())
            np.savez_compressed(
                artifact_path,
                event_delta=event_delta,
                normal_delta=normal_delta,
                key=np.asarray(key),
                label=np.asarray(label),
                hidden_length=np.asarray(hidden.shape[0], dtype=np.int64),
                score_length=np.asarray(raw_scores.size, dtype=np.int64),
                positive_count=np.asarray(positive.size, dtype=np.int64),
                background_count=np.asarray(background.size, dtype=np.int64),
                top_score_mean=np.asarray(aligned_scores[positive].mean(), dtype=np.float32),
                background_score_mean=np.asarray(aligned_scores[background].mean(), dtype=np.float32),
                positive_weight_mean=np.asarray(positive_weights.mean(), dtype=np.float32),
                temporal_support_mean=np.asarray(support_mean, dtype=np.float32),
                positive_rank_mean=np.asarray(ranks[positive].mean(), dtype=np.float32),
            )
            row = [
                key, label, int(hidden.shape[0]), int(raw_scores.size), int(positive.size), int(background.size),
                float(aligned_scores[positive].mean()), float(aligned_scores[background].mean()),
                float(positive_weights.mean()), support_mean, "computed",
            ]
        event_deltas.append(event_delta)
        normal_deltas.append(normal_delta)
        rows.append(row)
    if len(event_deltas) < 2:
        raise RuntimeError(f"only {len(event_deltas)} usable abnormal-video contributions; need at least two")

    event_array = np.stack(event_deltas, axis=0)
    normal_array = np.stack(normal_deltas, axis=0)
    event_mean, normal_mean_delta = event_array.mean(axis=0), normal_array.mean(axis=0)
    event_std = event_array.std(axis=0, ddof=1)
    normal_std_delta = normal_array.std(axis=0, ddof=1)
    event_score = np.abs(event_mean) / (event_std + args.sigma_min)
    normal_score = np.abs(normal_mean_delta) / (normal_std_delta + args.sigma_min)
    # A neuron is useful only if both localisation-oriented and normal-reference
    # effects are stable across abnormal training videos.
    selection_scores = np.minimum(event_score, normal_score).astype(np.float32)
    if not np.isfinite(selection_scores).all():
        raise RuntimeError("non-finite temporal-local selection score")

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
                "event_scores": event_score[layer, dims].astype(float).tolist(),
                "normal_scores": normal_score[layer, dims].astype(float).tolist(),
                "event_mean_deltas": event_mean[layer, dims].astype(float).tolist(),
                "normal_mean_deltas": normal_mean_delta[layer, dims].astype(float).tolist(),
                "event_std_deltas": event_std[layer, dims].astype(float).tolist(),
                "normal_std_deltas": normal_std_delta[layer, dims].astype(float).tolist(),
                "directions": np.sign(normal_mean_delta[layer, dims]).astype(int).tolist(),
            })
    selected_width = sum(len(item["dims"]) for item in selected)
    if selected_width != args.topk_global:
        raise RuntimeError(f"selected width={selected_width}, expected {args.topk_global}")

    artifact_paths = {
        "event_mean_delta_path": out_dir / "event_mean_delta.npy",
        "event_std_delta_path": out_dir / "event_std_delta.npy",
        "normal_mean_delta_path": out_dir / "normal_mean_delta.npy",
        "normal_std_delta_path": out_dir / "normal_std_delta.npy",
        "event_scores_path": out_dir / "event_scores.npy",
        "normal_scores_path": out_dir / "normal_scores.npy",
        "selection_scores_path": out_dir / "selection_scores.npy",
    }
    np.save(artifact_paths["event_mean_delta_path"], event_mean.astype(np.float32))
    np.save(artifact_paths["event_std_delta_path"], event_std.astype(np.float32))
    np.save(artifact_paths["normal_mean_delta_path"], normal_mean_delta.astype(np.float32))
    np.save(artifact_paths["normal_std_delta_path"], normal_std_delta.astype(np.float32))
    np.save(artifact_paths["event_scores_path"], event_score.astype(np.float32))
    np.save(artifact_paths["normal_scores_path"], normal_score.astype(np.float32))
    np.save(artifact_paths["selection_scores_path"], selection_scores)
    write_csv(
        out_dir / "per_video_contributions.csv",
        [
            "key", "label", "hidden_length", "raw_score_length", "positive_count", "background_count",
            "top_score_mean", "background_score_mean", "positive_weight_mean", "temporal_support_mean", "status",
        ],
        rows,
    )
    write_csv(out_dir / "skipped_videos.csv", ["key", "label", "role", "reason"], skipped)
    save_json(output_json, {
        "method": "vadclip_temporal_local_pure_normal_global_shift_v1",
        "dataset": args.dataset,
        "description": (
            "Per abnormal video: temporally supported top pseudo-score hidden mean minus distant lower-score "
            "same-video background, jointly ranked with its pure-normal z-score separation."
        ),
        "positive_definition": "top pseudo-score snippets in each abnormal training video, weighted by local score-rank support",
        "background_definition": "lowest-score eligible snippets outside each pseudo-positive temporal neighborhood in the same abnormal training video",
        "negative_definition": "pure normal training-video reference distribution represented by zero after normal z-scoring",
        "top_p": float(args.top_p),
        "background_p": float(args.background_p),
        "context_radius": int(args.context_radius),
        "normal_videos_used_for_reference": True,
        "abnormal_lower_score_snippets_used_as_same_video_background": True,
        "same_video_background_used_for_localisation": True,
        "frame_labels_used": False,
        "test_data_used": False,
        "normal_stat_snippets_per_video": int(args.normal_stat_snippets_per_video),
        "sigma_min": float(args.sigma_min),
        "num_normal_videos_for_reference": len(normal_keys),
        "num_normal_snippets_for_reference": int(normal_count),
        "num_abnormal_videos_with_contributions": len(event_deltas),
        "skipped_training_videos": len(skipped),
        "num_layers": int(selection_scores.shape[0]),
        "hidden_dim": int(selection_scores.shape[1]),
        "token_pool": token_pool,
        "selection_mode": "global",
        "topk_global": int(args.topk_global),
        "visual_width": int(args.topk_global),
        "selection_score": "minimum_of_absolute_cross_video_effect_sizes_for_temporal_local_and_pure_normal_deltas",
        "normal_mean_path": str(out_dir / "normal_mean.npy"),
        "normal_std_path": str(out_dir / "normal_std.npy"),
        "per_video_contribution_dir": str(contribution_dir),
        **{name: str(path) for name, path in artifact_paths.items()},
        "selected": selected,
    })
    print(
        f"wrote {output_json}: global top-{args.topk_global}; "
        f"positive=abnormal top-{args.top_p:.3f} with temporal support, "
        f"background=lowest {args.background_p:.3f} of distant same-video snippets, "
        "normal-reference agreement required",
        flush=True,
    )


if __name__ == "__main__":
    main()
