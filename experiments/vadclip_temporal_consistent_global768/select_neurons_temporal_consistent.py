#!/usr/bin/env python3
"""Select global neurons that are both abnormal and temporally well-localised.

For each abnormal training video this selector records two quantities per
hidden neuron:

* semantic shift: the z-scored hidden mean at frozen-baseline top-p snippets;
* temporal consistency: Spearman correlation between the whole hidden trace
  and the frozen-baseline score trace inside the same video.

Pure normal training videos define the z-score reference.  No abnormal-video
bottom snippets, test list, frame annotation, optimiser, or model checkpoint
are read.  A candidate must have a stable semantic shift *and* a stable
score-aligned temporal correlation across abnormal training videos.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


def add_injection_source() -> None:
    """Use only repository-local VadCLIP experiment utilities."""
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
    """Require the one hidden-token pooling contract used by concat building."""
    frame = pd.read_csv(path)
    pools = set(frame["token_pool"].astype(str)) if "token_pool" in frame.columns else {"cls"}
    if pools - {"cls", "patch_mean"} or len(pools) != 1:
        raise ValueError(f"{path}: expected exactly one valid token_pool, got {sorted(pools)}")
    return next(iter(pools))


def source_labels(path: str) -> dict[str, str]:
    """Map each source-training video key to its stable video label."""
    labels: dict[str, str] = {}
    for _, row in read_csv(path).iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"{path}: video {key!r} has inconsistent labels")
        labels[key] = label
    return labels


def pseudo_score_map(path: str) -> dict[str, tuple[str, str]]:
    """Read frozen-baseline pseudo scores written by score_vadclip_pseudo.py."""
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
    """Estimate per-neuron pure-normal mean/std with equal-capped videos."""
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
    """Reuse a complete normal-stat cache or create it transactionally."""
    cache_path = output_dir / "normal_stats.npz"
    mean_path, std_path = output_dir / "normal_mean.npy", output_dir / "normal_std.npy"
    if cache_path.is_file() and not no_resume:
        with np.load(cache_path, allow_pickle=False) as artifact:
            required = {"mean", "std", "snippet_count", "limit_per_video"}
            if not required.issubset(artifact.files):
                raise ValueError(f"{cache_path}: incomplete normal-stat artifact; use --no-resume or --clean")
            mean = np.asarray(artifact["mean"], dtype=np.float32)
            std = np.asarray(artifact["std"], dtype=np.float32)
            count = int(artifact["snippet_count"].item())
            saved_limit = int(artifact["limit_per_video"].item())
        if saved_limit != int(limit_per_video):
            raise ValueError(
                f"{cache_path}: normal-stat limit={saved_limit} differs from "
                f"--normal-stat-snippets-per-video={limit_per_video}; use --clean"
            )
        if mean.ndim != 2 or std.shape != mean.shape or count < 2:
            raise ValueError(f"{cache_path}: invalid normal-stat artifact; use --no-resume or --clean")
        if not mean_path.is_file():
            np.save(mean_path, mean)
        if not std_path.is_file():
            np.save(std_path, std)
        print(f"reuse pure-normal statistics: {cache_path}", flush=True)
        return mean, std, count

    mean, std, count = collect_normal_stats(normal_paths, limit_per_video)
    np.savez_compressed(
        cache_path,
        mean=mean,
        std=std,
        snippet_count=np.asarray(count, dtype=np.int64),
        limit_per_video=np.asarray(limit_per_video, dtype=np.int64),
    )
    np.save(mean_path, mean)
    np.save(std_path, std)
    print(f"wrote pure-normal statistics from {count} snippets", flush=True)
    return mean, std, count


def top_indices(scores: np.ndarray, top_p: float) -> np.ndarray:
    """Match the established global-768 top-count convention, without bottom samples."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError(f"need at least two pseudo scores, got {values.size}")
    requested = max(1, int(np.ceil(float(top_p) * values.size)))
    count = min(requested, values.size // 2)
    return np.argsort(values, kind="mergesort")[-count:][::-1].astype(np.int64)


def average_rank(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks for a one-dimensional score sequence."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("cannot rank an empty or non-finite pseudo-score sequence")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranked = np.empty(values.size, dtype=np.float32)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranked[order[start:stop]] = (start + stop - 1) * 0.5
        start = stop
    return ranked


def ordinal_spearman(hidden: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Compute per-neuron within-video Spearman correlation without SciPy.

    Hidden tensors are floating-point activations, where ties are practically
    absent.  We still use a stable ordinal ranking for deterministic behaviour;
    pseudo-score ties receive their exact average ranks.  Constant hidden
    traces and constant pseudo-score traces intentionally contribute zero.
    """
    if hidden.ndim != 3 or hidden.shape[0] != scores.size:
        raise ValueError(f"hidden/scores must align as [T,L,D]/[T], got {hidden.shape}/{scores.shape}")
    length, layers, hidden_dim = hidden.shape
    if length < 2:
        return np.zeros((layers, hidden_dim), dtype=np.float32)
    score_rank = average_rank(scores)
    score_centered = score_rank - score_rank.mean()
    score_norm = float(np.linalg.norm(score_centered))
    if score_norm <= 1e-8:
        return np.zeros((layers, hidden_dim), dtype=np.float32)

    flat = np.asarray(hidden, dtype=np.float32).reshape(length, -1)
    order = np.argsort(flat, axis=0, kind="mergesort")
    ranks = np.empty_like(flat, dtype=np.float32)
    columns = np.arange(flat.shape[1], dtype=np.int64)[None, :]
    ranks[order, columns] = np.arange(length, dtype=np.float32)[:, None]
    ranks -= ranks.mean(axis=0, keepdims=True)
    hidden_norm = np.linalg.norm(ranks, axis=0)
    correlation = (score_centered @ ranks) / np.maximum(score_norm * hidden_norm, 1e-8)
    constant = np.ptp(flat, axis=0) <= 1e-8
    correlation[constant] = 0.0
    return np.clip(correlation.reshape(layers, hidden_dim), -1.0, 1.0).astype(np.float32)


def contribution_artifact_path(contribution_dir: Path, key: str) -> Path:
    """One abnormal-video artifact makes the expensive ranking stage resumable."""
    return contribution_dir / f"{key}.npz"


def load_contribution(
    path: Path,
    key: str,
    label: str,
    expected_shape: tuple[int, int],
    top_p: float,
) -> tuple[np.ndarray, np.ndarray, list[object]]:
    """Validate one completed video contribution before it is reused."""
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "semantic_delta", "temporal_correlation", "key", "label", "hidden_length",
            "score_length", "top_count", "top_score_mean", "top_p",
        }
        if not required.issubset(artifact.files):
            raise ValueError(f"{path}: incomplete contribution artifact; use --no-resume or --clean")
        if str(artifact["key"].item()) != key or str(artifact["label"].item()) != label:
            raise ValueError(f"{path}: key/label differs from current source CSV; use --clean")
        saved_top_p = float(artifact["top_p"].item())
        if not np.isclose(saved_top_p, top_p, rtol=0.0, atol=1e-12):
            raise ValueError(f"{path}: top-p={saved_top_p} differs from current --top-p={top_p}; use --clean")
        semantic_delta = np.asarray(artifact["semantic_delta"], dtype=np.float32)
        temporal_correlation = np.asarray(artifact["temporal_correlation"], dtype=np.float32)
        row = [
            key,
            label,
            int(artifact["hidden_length"].item()),
            int(artifact["score_length"].item()),
            int(artifact["top_count"].item()),
            float(artifact["top_score_mean"].item()),
            float(np.mean(np.abs(temporal_correlation))),
            "reused",
        ]
    for name, value in (("semantic_delta", semantic_delta), ("temporal_correlation", temporal_correlation)):
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid {name} shape/value; use --no-resume or --clean")
    return semantic_delta, temporal_correlation, row


def global_top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    """Return deterministic global top-k flattened layer/dimension positions."""
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if topk <= 0 or topk > flat.size:
        raise ValueError(f"topk={topk} must be in [1, {flat.size}]")
    return np.argsort(-flat, kind="mergesort")[:topk]


def selection_components(
    semantic_deltas: np.ndarray,
    temporal_correlations: np.ndarray,
    sigma_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine semantic and time-localisation evidence with cross-video stability.

    The direction is inferred from the mean semantic shift.  A neuron shifted
    upward at pseudo-positive snippets should have positive score correlation;
    a downward-shifted neuron should have negative correlation.  The latter is
    therefore direction-aligned before its effect size is calculated.
    """
    if semantic_deltas.shape != temporal_correlations.shape or semantic_deltas.ndim != 3:
        raise ValueError("expected matching [video, layer, hidden] contribution arrays")
    if semantic_deltas.shape[0] < 2:
        raise ValueError("need at least two abnormal videos for cross-video stability")
    semantic_mean = semantic_deltas.mean(axis=0)
    semantic_std = semantic_deltas.std(axis=0, ddof=1)
    semantic_effect = np.abs(semantic_mean) / (semantic_std + sigma_min)
    direction = np.sign(semantic_mean)
    direction[direction == 0.0] = 1.0
    aligned_temporal = temporal_correlations * direction[None, :, :]
    temporal_mean = aligned_temporal.mean(axis=0)
    temporal_std = aligned_temporal.std(axis=0, ddof=1)
    temporal_effect = np.maximum(temporal_mean, 0.0) / (temporal_std + sigma_min)
    combined = semantic_effect * temporal_effect
    if not np.isfinite(combined).all():
        raise RuntimeError("non-finite temporal-consistent selection score")
    return (
        semantic_mean.astype(np.float32),
        semantic_std.astype(np.float32),
        temporal_mean.astype(np.float32),
        temporal_std.astype(np.float32),
        semantic_effect.astype(np.float32),
        combined.astype(np.float32),
    )


def bootstrap_frequency(
    semantic_deltas: np.ndarray,
    temporal_correlations: np.ndarray,
    topk: int,
    sigma_min: float,
    rounds: int,
    seed: int,
) -> np.ndarray:
    """Estimate global-top-k inclusion frequency by video-level bootstrap."""
    shape = tuple(semantic_deltas.shape[1:])
    if rounds == 0:
        return np.ones(shape, dtype=np.float32)
    generator = np.random.default_rng(seed)
    hits = np.zeros(shape, dtype=np.int32)
    count = semantic_deltas.shape[0]
    probabilities = np.full(count, 1.0 / float(count), dtype=np.float64)
    for _ in tqdm(range(rounds), desc="bootstrap temporal-consistent stability", unit="round"):
        # Multinomial counts are exactly equivalent to sampling video indices
        # with replacement, but avoid duplicating the full [V,L,D] tensors.
        weights = generator.multinomial(count, probabilities).astype(np.float32)
        view = weights[:, None, None]
        semantic_mean = (semantic_deltas * view).sum(axis=0) / float(count)
        semantic_second = (np.square(semantic_deltas) * view).sum(axis=0)
        semantic_std = np.sqrt(np.maximum((semantic_second - count * np.square(semantic_mean)) / (count - 1), 0.0))
        semantic_effect = np.abs(semantic_mean) / (semantic_std + sigma_min)
        direction = np.sign(semantic_mean)
        direction[direction == 0.0] = 1.0
        temporal_mean = (temporal_correlations * view).sum(axis=0) / float(count)
        temporal_second = (np.square(temporal_correlations) * view).sum(axis=0)
        temporal_std = np.sqrt(np.maximum((temporal_second - count * np.square(temporal_mean)) / (count - 1), 0.0))
        temporal_effect = np.maximum(temporal_mean * direction, 0.0) / (temporal_std + sigma_min)
        combined = semantic_effect * temporal_effect
        if not np.isfinite(combined).all():
            raise RuntimeError("non-finite bootstrap temporal-consistent selection score")
        hits.reshape(-1)[global_top_indices(combined, topk)] += 1
    return (hits.astype(np.float32) / float(rounds)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select global neurons using top-vs-normal semantics and within-video temporal consistency."
    )
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--topk-global", type=int, default=768)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--bootstrap-rounds", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5]")
    if args.topk_global <= 0 or args.normal_stat_snippets_per_video <= 0 or args.sigma_min <= 0.0:
        parser.error("topk-global, normal-stat-snippets-per-video, and sigma-min must be positive")
    if args.bootstrap_rounds < 0:
        parser.error("--bootstrap-rounds must be non-negative")
    for path in (args.source_train_csv, args.hidden_manifest, args.pseudo_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing temporal-consistent selection input: {path}")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    output_json = out_dir / "selected_neurons.json"
    if output_json.is_file() and not args.no_resume:
        print(f"reuse completed temporal-consistent global selection: {output_json}", flush=True)
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
        f"temporally-audited abnormal videos={len(abnormal_keys)}",
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
    semantic_deltas, temporal_correlations, rows = [], [], []
    for key in tqdm(abnormal_keys, desc="temporal-consistent video contributions", unit="video"):
        label = labels_by_key[key]
        artifact_path = contribution_artifact_path(contribution_dir, key)
        if artifact_path.is_file() and not args.no_resume:
            semantic_delta, temporal_correlation, row = load_contribution(
                artifact_path, key, label, expected_shape, args.top_p
            )
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
            semantic_delta = z_hidden[positive].mean(axis=0).astype(np.float32)
            temporal_correlation = ordinal_spearman(z_hidden, aligned_scores)
            if not np.isfinite(semantic_delta).all() or not np.isfinite(temporal_correlation).all():
                raise RuntimeError(f"{key}: non-finite semantic or temporal contribution")
            np.savez_compressed(
                artifact_path,
                semantic_delta=semantic_delta,
                temporal_correlation=temporal_correlation,
                key=np.asarray(key),
                label=np.asarray(label),
                hidden_length=np.asarray(hidden.shape[0], dtype=np.int64),
                score_length=np.asarray(raw_scores.size, dtype=np.int64),
                top_count=np.asarray(positive.size, dtype=np.int64),
                top_score_mean=np.asarray(aligned_scores[positive].mean(), dtype=np.float32),
                top_p=np.asarray(args.top_p, dtype=np.float64),
            )
            row = [
                key,
                label,
                int(hidden.shape[0]),
                int(raw_scores.size),
                int(positive.size),
                float(aligned_scores[positive].mean()),
                float(np.mean(np.abs(temporal_correlation))),
                "computed",
            ]
        semantic_deltas.append(semantic_delta)
        temporal_correlations.append(temporal_correlation)
        rows.append(row)
    if len(semantic_deltas) < 2:
        raise RuntimeError(f"only {len(semantic_deltas)} usable abnormal-video contributions; need at least two")

    semantic_array = np.stack(semantic_deltas, axis=0)
    temporal_array = np.stack(temporal_correlations, axis=0)
    (
        semantic_mean,
        semantic_std,
        temporal_mean,
        temporal_std,
        semantic_effect,
        raw_selection_scores,
    ) = selection_components(semantic_array, temporal_array, args.sigma_min)
    stability_frequency = bootstrap_frequency(
        semantic_array,
        temporal_array,
        args.topk_global,
        args.sigma_min,
        args.bootstrap_rounds,
        args.bootstrap_seed,
    )
    selection_scores = raw_selection_scores * stability_frequency
    if not np.isfinite(selection_scores).all():
        raise RuntimeError("non-finite bootstrap-stabilised selection score")
    flat = global_top_indices(selection_scores, args.topk_global)
    hidden_dim = selection_scores.shape[1]
    directions = np.sign(semantic_mean).astype(np.int8)
    directions[directions == 0] = 1
    selected = []
    for layer in range(selection_scores.shape[0]):
        dims = (flat[flat // hidden_dim == layer] % hidden_dim).astype(np.int64)
        if dims.size:
            selected.append({
                "layer_index": int(layer),
                "dims": dims.tolist(),
                "scores": selection_scores[layer, dims].astype(float).tolist(),
                "semantic_effects": semantic_effect[layer, dims].astype(float).tolist(),
                "semantic_mean_deltas": semantic_mean[layer, dims].astype(float).tolist(),
                "semantic_std_deltas": semantic_std[layer, dims].astype(float).tolist(),
                "temporal_aligned_means": temporal_mean[layer, dims].astype(float).tolist(),
                "temporal_aligned_stds": temporal_std[layer, dims].astype(float).tolist(),
                "bootstrap_frequencies": stability_frequency[layer, dims].astype(float).tolist(),
                "directions": directions[layer, dims].astype(int).tolist(),
            })
    selected_width = sum(len(item["dims"]) for item in selected)
    if selected_width != args.topk_global:
        raise RuntimeError(f"selected width={selected_width}, expected {args.topk_global}")

    semantic_mean_path = out_dir / "semantic_mean_delta.npy"
    semantic_std_path = out_dir / "semantic_std_delta.npy"
    temporal_mean_path = out_dir / "temporal_aligned_mean.npy"
    temporal_std_path = out_dir / "temporal_aligned_std.npy"
    semantic_effect_path = out_dir / "semantic_effect_scores.npy"
    raw_score_path = out_dir / "raw_selection_scores.npy"
    stability_path = out_dir / "bootstrap_frequency.npy"
    score_path = out_dir / "selection_scores.npy"
    np.save(semantic_mean_path, semantic_mean)
    np.save(semantic_std_path, semantic_std)
    np.save(temporal_mean_path, temporal_mean)
    np.save(temporal_std_path, temporal_std)
    np.save(semantic_effect_path, semantic_effect)
    np.save(raw_score_path, raw_selection_scores)
    np.save(stability_path, stability_frequency)
    np.save(score_path, selection_scores.astype(np.float32))
    write_csv(
        out_dir / "per_video_contributions.csv",
        [
            "key", "label", "hidden_length", "raw_score_length", "top_count", "top_score_mean",
            "mean_absolute_temporal_spearman", "status",
        ],
        rows,
    )
    write_csv(out_dir / "skipped_videos.csv", ["key", "label", "role", "reason"], skipped)
    save_json(output_json, {
        "method": "vadclip_top_vs_normal_temporal_consistent_global768_v1",
        "description": (
            "Globally rank neurons by the product of stable top-vs-pure-normal semantic shift "
            "and stable direction-aligned within-video Spearman correlation with frozen-baseline scores."
        ),
        "dataset": args.dataset,
        "positive_definition": "top frozen-baseline pseudo-score snippets within each abnormal training video",
        "negative_definition": "pure-normal training-video reference distribution, represented by zero after normal z-scoring",
        "temporal_definition": "within each abnormal training video, stable ordinal Spearman(hidden trace, frozen-baseline score trace)",
        "semantic_direction_rule": "positive semantic shift requires positive correlation; negative shift requires negative correlation",
        "top_p": float(args.top_p),
        "normal_videos_used_for_reference": True,
        "abnormal_bottom_snippets_used_as_negatives": False,
        "frame_labels_used": False,
        "test_data_used": False,
        "normal_stat_snippets_per_video": int(args.normal_stat_snippets_per_video),
        "sigma_min": float(args.sigma_min),
        "num_normal_videos_for_reference": len(normal_keys),
        "num_normal_snippets_for_reference": int(normal_count),
        "num_abnormal_videos_with_contributions": len(semantic_deltas),
        "skipped_training_videos": len(skipped),
        "num_layers": int(selection_scores.shape[0]),
        "hidden_dim": int(selection_scores.shape[1]),
        "token_pool": token_pool,
        "selection_mode": "global",
        "topk_global": int(args.topk_global),
        "visual_width": int(args.topk_global),
        "selection_score": "semantic_effect_size_times_positive_direction_aligned_temporal_spearman_effect_size_times_bootstrap_frequency",
        "bootstrap_rounds": int(args.bootstrap_rounds),
        "bootstrap_seed": int(args.bootstrap_seed),
        "normal_mean_path": str(out_dir / "normal_mean.npy"),
        "normal_std_path": str(out_dir / "normal_std.npy"),
        "semantic_mean_delta_path": str(semantic_mean_path),
        "semantic_std_delta_path": str(semantic_std_path),
        "temporal_aligned_mean_path": str(temporal_mean_path),
        "temporal_aligned_std_path": str(temporal_std_path),
        "semantic_effect_scores_path": str(semantic_effect_path),
        "raw_selection_scores_path": str(raw_score_path),
        "bootstrap_frequency_path": str(stability_path),
        "selection_scores_path": str(score_path),
        "per_video_contribution_dir": str(contribution_dir),
        "selected": selected,
    })
    print(
        f"wrote {output_json}: global top-{args.topk_global}; "
        f"top-vs-normal semantics + temporal consistency + {args.bootstrap_rounds} bootstrap rounds",
        flush=True,
    )


if __name__ == "__main__":
    main()
