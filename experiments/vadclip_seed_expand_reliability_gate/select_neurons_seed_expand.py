#!/usr/bin/env python3
"""Select global neurons from reliable expanded pseudo-positive snippets.

Unlike the original top-vs-bottom rule, this selector never treats low-score
snippets inside an abnormal video as negative.  It contrasts a soft,
high-precision expanded seed set with the pure-normal training distribution.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def add_injection_source() -> None:
    source = str(Path(__file__).resolve().parents[1] / "vadclip_neuron_injection")
    if source not in sys.path:
        sys.path.insert(0, source)


add_injection_source()
from common import clean_dir, ensure_dir, is_normal_label, load_hidden, save_json, write_csv  # noqa: E402
from reliability import (  # noqa: E402
    bounded_top_indices,
    build_config,
    collect_normal_hidden_stats,
    config_as_dict,
    labels_by_key,
    manifest_map,
    manifest_token_pool,
    pseudo_score_paths,
    reliability_map,
)


def global_top_indices(scores: np.ndarray, topk: int) -> np.ndarray:
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not 0 < topk <= flat.size:
        raise ValueError(f"topk-global={topk} must be in [1, {flat.size}]")
    return np.argsort(-flat, kind="mergesort")[:topk]


def artifact_path(directory: Path, key: str) -> Path:
    return directory / f"{key}.npz"


def load_delta(path: Path, key: str, label: str, shape: tuple[int, int]) -> tuple[np.ndarray, list[object]]:
    with np.load(path, allow_pickle=False) as artifact:
        required = {
            "delta", "key", "label", "hidden_length", "score_length", "seed_count", "expanded_count",
            "weight_sum", "reliability_mean", "reliability_max",
        }
        if not required.issubset(artifact.files):
            raise ValueError(f"{path}: incomplete resume artifact; use --no-resume or --clean")
        if str(artifact["key"].item()) != key or str(artifact["label"].item()) != label:
            raise ValueError(f"{path}: current train CSV differs; use --clean")
        delta = np.asarray(artifact["delta"], dtype=np.float32)
        row = [
            key, label, int(artifact["hidden_length"].item()), int(artifact["score_length"].item()),
            int(artifact["seed_count"].item()), int(artifact["expanded_count"].item()),
            float(artifact["weight_sum"].item()), float(artifact["reliability_mean"].item()),
            float(artifact["reliability_max"].item()), "reused",
        ]
    if delta.shape != shape or not np.isfinite(delta).all():
        raise ValueError(f"{path}: invalid delta; use --no-resume or --clean")
    return delta, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Select global neurons with high-precision seed expansion.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True, help="Frozen VadCLIP classifier-probability group_scores.csv.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed-top-p", type=float, default=0.10)
    parser.add_argument("--expand-top-p", type=float, default=0.30)
    parser.add_argument("--topk-global", type=int, default=768)
    parser.add_argument("--normal-score-quantile", type=float, default=0.95)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--normal-score-snippets-per-video", type=int, default=256)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.topk_global <= 0 or args.normal_stat_snippets_per_video <= 0:
        parser.error("topk-global and normal-stat-snippets-per-video must be positive")
    for path in (args.source_train_csv, args.hidden_manifest, args.pseudo_csv):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing seed-expansion selector input: {path}")
    if args.clean and args.no_resume:
        parser.error("--clean and --no-resume cannot be used together")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    output_json = out_dir / "selected_neurons.json"
    if output_json.is_file() and not args.no_resume:
        print(f"reuse completed seed-expansion selection: {output_json}", flush=True)
        return

    hidden_paths = manifest_map(args.hidden_manifest)
    token_pool = manifest_token_pool(args.hidden_manifest)
    labels = labels_by_key(args.source_train_csv)
    scores = pseudo_score_paths(args.pseudo_csv)
    normal_keys, abnormal_keys, skipped = [], [], []
    for key, label in sorted(labels.items()):
        role = "normal" if is_normal_label(args.dataset, label) else "abnormal"
        if key not in hidden_paths:
            skipped.append([key, label, role, "missing_hidden"])
        elif role == "normal":
            normal_keys.append(key)
        elif key not in scores:
            skipped.append([key, label, role, "missing_pseudo_score"])
        elif scores[key][0] != label:
            raise ValueError(f"{key}: pseudo label {scores[key][0]!r} differs from source label {label!r}")
        else:
            abnormal_keys.append(key)
    if not normal_keys or len(abnormal_keys) < 2:
        raise RuntimeError(f"need normal and at least two abnormal videos; got normal={len(normal_keys)}, abnormal={len(abnormal_keys)}")

    config, score_audit = build_config(
        args.dataset, args.source_train_csv, args.pseudo_csv, args.seed_top_p, args.expand_top_p,
        args.normal_score_quantile, args.score_temperature, args.sigma_min,
        args.normal_score_snippets_per_video,
    )
    normal_mean, normal_std, normal_count = collect_normal_hidden_stats(
        [hidden_paths[key] for key in normal_keys], args.normal_stat_snippets_per_video
    )
    np.save(out_dir / "normal_mean.npy", normal_mean)
    np.save(out_dir / "normal_std.npy", normal_std)
    save_json(out_dir / "reliability_config.json", {
        "method": "vadclip_seed_expand_reliability_v1",
        **config_as_dict(config),
        **score_audit,
        "source_train_csv": args.source_train_csv,
        "pseudo_csv": args.pseudo_csv,
        "normal_stat_snippets_per_video": int(args.normal_stat_snippets_per_video),
        "normal_hidden_snippets": int(normal_count),
    })
    print(
        f"matched pure-normal={len(normal_keys)}, abnormal={len(abnormal_keys)}; "
        f"normal score q{config.normal_score_quantile:.2f}={config.normal_score_threshold:.6f}",
        flush=True,
    )

    delta_dir = ensure_dir(out_dir / "per_video_deltas")
    expected_shape = tuple(normal_mean.shape)
    deltas, rows = [], []
    for key in tqdm(abnormal_keys, desc="reliability-weighted video deltas", unit="video"):
        label, output_path = labels[key], artifact_path(delta_dir, key)
        if output_path.is_file() and not args.no_resume:
            delta, row = load_delta(output_path, key, label, expected_shape)
        else:
            hidden, _metadata = load_hidden(hidden_paths[key])
            if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected_shape or hidden.shape[0] == 0:
                raise ValueError(f"{key}: hidden shape {hidden.shape} differs from normal statistics {expected_shape}")
            raw_scores = np.asarray(np.load(scores[key][1], allow_pickle=False), dtype=np.float32).reshape(-1)
            if raw_scores.size == 0 or not np.isfinite(raw_scores).all():
                raise ValueError(f"{scores[key][1]}: expected finite non-empty score array")
            q, aligned, seeds, _similarity = reliability_map(hidden, raw_scores, normal_mean, normal_std, config)
            expanded = bounded_top_indices(q, config.expand_top_p)
            weights = q[expanded]
            if float(weights.sum()) <= 1e-8:
                # With finite scores/similarities this can only happen for an
                # extreme calibration mismatch.  It must be visible, never a
                # silent all-zero video contribution.
                raise RuntimeError(f"{key}: expanded reliability weights sum to zero")
            z_hidden = (hidden - normal_mean) / (normal_std + config.sigma_min)
            delta = np.average(z_hidden[expanded], axis=0, weights=weights).astype(np.float32)
            np.savez_compressed(
                output_path, delta=delta, key=np.asarray(key), label=np.asarray(label),
                hidden_length=np.asarray(hidden.shape[0], dtype=np.int64),
                score_length=np.asarray(raw_scores.size, dtype=np.int64),
                seed_count=np.asarray(seeds.size, dtype=np.int64),
                expanded_count=np.asarray(expanded.size, dtype=np.int64),
                weight_sum=np.asarray(weights.sum(), dtype=np.float32),
                reliability_mean=np.asarray(q[expanded].mean(), dtype=np.float32),
                reliability_max=np.asarray(q.max(), dtype=np.float32),
                expanded_indices=expanded.astype(np.int64), aligned_scores=aligned.astype(np.float32), reliability=q,
            )
            row = [
                key, label, int(hidden.shape[0]), int(raw_scores.size), int(seeds.size), int(expanded.size),
                float(weights.sum()), float(q[expanded].mean()), float(q.max()), "computed",
            ]
        deltas.append(delta)
        rows.append(row)

    delta_array = np.stack(deltas, axis=0)
    mean_delta = delta_array.mean(axis=0)
    std_delta = delta_array.std(axis=0, ddof=1)
    selection_scores = np.abs(mean_delta) / (std_delta + config.sigma_min)
    if not np.isfinite(selection_scores).all():
        raise RuntimeError("non-finite seed-expansion selection scores")
    chosen = global_top_indices(selection_scores, args.topk_global)
    hidden_dim = selection_scores.shape[1]
    selected = []
    for layer in range(selection_scores.shape[0]):
        dims = (chosen[chosen // hidden_dim == layer] % hidden_dim).astype(np.int64)
        if dims.size:
            selected.append({
                "layer_index": int(layer), "dims": dims.tolist(),
                "scores": selection_scores[layer, dims].astype(float).tolist(),
                "mean_deltas": mean_delta[layer, dims].astype(float).tolist(),
                "std_deltas": std_delta[layer, dims].astype(float).tolist(),
                "directions": np.sign(mean_delta[layer, dims]).astype(int).tolist(),
            })
    if sum(len(item["dims"]) for item in selected) != args.topk_global:
        raise RuntimeError("global neuron selection width does not match topk-global")
    np.save(out_dir / "mean_delta.npy", mean_delta.astype(np.float32))
    np.save(out_dir / "std_delta.npy", std_delta.astype(np.float32))
    np.save(out_dir / "selection_scores.npy", selection_scores.astype(np.float32))
    write_csv(
        out_dir / "per_video_reliability_rows.csv",
        ["key", "label", "hidden_length", "raw_score_length", "seed_count", "expanded_count", "weight_sum", "expanded_reliability_mean", "reliability_max", "status"],
        rows,
    )
    write_csv(out_dir / "skipped_videos.csv", ["key", "label", "role", "reason"], skipped)
    save_json(output_json, {
        "method": "vadclip_seed_expand_reliability_global_shift_v1",
        "description": "Per abnormal video, reliability-weighted expanded seed hidden mean minus pure-normal reference; global rank by absolute cross-video effect size.",
        "positive_definition": "top reliability snippets, where reliability requires an abnormal normal-calibrated frozen score and cosine agreement with the video top-score seed prototype",
        "negative_definition": "pure-normal training distribution represented by zero after z-scoring",
        "abnormal_bottom_snippets_used_as_negatives": False,
        "frame_labels_used": False,
        "test_data_used": False,
        "dataset": args.dataset,
        **config_as_dict(config),
        **score_audit,
        "num_normal_videos_for_reference": len(normal_keys),
        "num_normal_snippets_for_reference": int(normal_count),
        "num_abnormal_videos_with_reliability_deltas": len(deltas),
        "skipped_training_videos": len(skipped),
        "normal_mean_path": str(out_dir / "normal_mean.npy"),
        "normal_std_path": str(out_dir / "normal_std.npy"),
        "token_pool": token_pool,
        "selection_mode": "global",
        "topk_global": int(args.topk_global),
        "visual_width": int(args.topk_global),
        "hidden_dim": int(selection_scores.shape[1]),
        "num_layers": int(selection_scores.shape[0]),
        "selection_score": "absolute_mean_reliability_expanded_vs_pure_normal_z_delta_over_cross_video_std",
        "selected": selected,
    })
    print(f"wrote {output_json}: reliability-expanded global top-{args.topk_global}", flush=True)


if __name__ == "__main__":
    main()
