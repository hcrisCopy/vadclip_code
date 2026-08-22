#!/usr/bin/env python3
"""Offline test-label diagnostic for label-free reliability seed expansion.

Frame labels are intentionally isolated in this script: they assess whether
the training-only reliability rule is worth a formal run, but cannot alter
training data, neuron selection, model weights, or checkpoint selection.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def add_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "vadclip_neuron_injection", root / "vadclip_pseudo_label_diagnostic"):
        source = str(directory)
        if source not in sys.path:
            sys.path.insert(0, source)


add_sources()
from common import base_key, clean_dir, ensure_dir, is_normal_label, load_clip_feature, read_csv, save_json  # noqa: E402
from analyze_pseudo_label_quality import (  # noqa: E402
    ScoreRecord,
    artifact_path,
    load_baseline,
    load_record,
    prompt_text,
    score_feature,
)
from reliability import (  # noqa: E402
    bounded_top_indices,
    build_config,
    collect_normal_hidden_stats,
    config_as_dict,
    labels_by_key,
    manifest_map,
    reliability_map,
    top_indices,
)


def subset_stats(frame_gt: np.ndarray, indices: np.ndarray) -> dict[str, int | float | None]:
    selected = frame_gt[indices]
    positives, frames, video_positive = int(selected.sum()), int(selected.size), int(frame_gt.sum())
    return {
        "snippet_count": int(indices.size), "frame_count": frames, "positive_frames": positives,
        "positive_rate": float(positives / frames) if frames else None,
        "positive_recall": float(positives / video_positive) if video_positive else None,
    }


def aggregate(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int | None]:
    abnormal = [row for row in rows if bool(row["is_abnormal"])]
    selected_frames = sum(int(row[f"{prefix}_frame_count"]) for row in abnormal)
    selected_positive = sum(int(row[f"{prefix}_positive_frames"]) for row in abnormal)
    total_positive = sum(int(row["positive_frames"]) for row in abnormal)
    macro_precision = [row[f"{prefix}_positive_rate"] for row in abnormal if row[f"{prefix}_positive_rate"] is not None]
    macro_recall = [row[f"{prefix}_positive_recall"] for row in abnormal if row[f"{prefix}_positive_recall"] is not None]
    return {
        "abnormal_videos": len(abnormal), "selected_frames": selected_frames,
        "selected_positive_frames": selected_positive,
        "micro_positive_rate": float(selected_positive / selected_frames) if selected_frames else None,
        "micro_positive_recall": float(selected_positive / total_positive) if total_positive else None,
        "macro_positive_rate": float(np.mean(macro_precision)) if macro_precision else None,
        "macro_positive_recall": float(np.mean(macro_recall)) if macro_recall else None,
    }


def aggregate_normal_reliability(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    normal = [row for row in rows if not bool(row["is_abnormal"])]
    values = [float(row["reliability_mean"]) for row in normal]
    maxima = [float(row["reliability_max"]) for row in normal]
    return {
        "normal_videos": len(normal),
        "mean_of_video_means": float(np.mean(values)) if values else None,
        "mean_of_video_maxima": float(np.mean(maxima)) if maxima else None,
        "p95_video_maximum": float(np.quantile(maxima, 0.95)) if maxima else None,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("no diagnostic rows were created")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure label-free reliability seed expansion quality with held-out frame GT.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--source-train-csv", required=True, help="Training CSV; supplies only normal labels for calibration.")
    parser.add_argument("--train-hidden-manifest", required=True, help="Training hidden cache; supplies only normal hidden statistics.")
    parser.add_argument("--pseudo-csv", required=True, help="Frozen baseline training group_scores.csv.")
    parser.add_argument("--test-list", required=True, help="Official 512D test CSV; never used for training.")
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--model-path", required=True, help="Frozen official VadCLIP checkpoint.")
    parser.add_argument("--gt-path", required=True, help="Frame labels used only in this offline diagnostic.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--seed-top-p", type=float, default=0.10)
    parser.add_argument("--expand-top-p", type=float, default=0.30)
    parser.add_argument("--normal-score-quantile", type=float, default=0.95)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--normal-score-snippets-per-video", type=int, default=256)
    parser.add_argument("--normal-stat-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.no_resume:
        parser.error("--clean and --no-resume cannot be used together")
    for path in (
        args.source_train_csv, args.train_hidden_manifest, args.pseudo_csv, args.test_list,
        args.test_hidden_manifest, args.model_path, args.gt_path,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing diagnostic input: {path}")

    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video_scores")
    labels = labels_by_key(args.source_train_csv)
    train_hidden = manifest_map(args.train_hidden_manifest)
    normal_paths = [train_hidden[key] for key, label in labels.items() if is_normal_label(args.dataset, label) and key in train_hidden]
    if not normal_paths:
        raise RuntimeError("no training pure-normal hidden artifacts matched the training CSV")
    config, score_audit = build_config(
        args.dataset, args.source_train_csv, args.pseudo_csv, args.seed_top_p, args.expand_top_p,
        args.normal_score_quantile, args.score_temperature, args.sigma_min,
        args.normal_score_snippets_per_video,
    )
    normal_mean, normal_std, normal_count = collect_normal_hidden_stats(normal_paths, args.normal_stat_snippets_per_video)
    test_hidden = manifest_map(args.test_hidden_manifest)
    test_frame = read_csv(args.test_list).reset_index(drop=True)
    gt = np.asarray(np.load(args.gt_path), dtype=np.int8).reshape(-1)
    if not np.isin(gt, [0, 1]).all():
        raise ValueError(f"{args.gt_path}: expected binary frame labels")
    save_json(out_dir / "run_config.json", {
        "method": "vadclip_seed_expand_reliability_quality_diagnostic_v1",
        "dataset": args.dataset, "source_train_csv": args.source_train_csv,
        "test_list": args.test_list, "model_path": args.model_path, "gt_path": args.gt_path,
        **config_as_dict(config), **score_audit,
        "normal_hidden_snippets": normal_count,
        "frame_labels_used_only_for_offline_diagnosis": True,
        "formal_training_or_checkpoint_selection_changed": False,
    })

    device = torch.device(args.device)
    model, options = load_baseline(args.dataset, args.vadclip_root, args.model_path, device)
    prompts = prompt_text(args.dataset)
    records: list[ScoreRecord] = []
    used_outputs: set[Path] = set()
    for _, row in tqdm(test_frame.iterrows(), total=len(test_frame), desc="frozen baseline scores", unit="video"):
        source_path, label = str(row["path"]), str(row["label"])
        output_path = artifact_path(score_dir, source_path)
        if output_path in used_outputs:
            raise ValueError(f"duplicate test feature stem: {output_path}")
        used_outputs.add(output_path)
        if output_path.is_file() and not args.no_resume:
            record = load_record(output_path, source_path, label)
        else:
            feature = load_clip_feature(source_path)
            if feature.shape[1] != options.visual_width:
                raise ValueError(f"{source_path}: expected official {options.visual_width}D feature")
            prob1, prob2 = score_feature(model, feature, options.visual_length, prompts, device)
            np.savez_compressed(output_path, prob1=prob1, prob2=prob2, source_path=np.asarray(source_path), label=np.asarray(label))
            record = ScoreRecord(source_path, label, prob1, prob2)
        records.append(record)
    expected_frames = sum(record.prob1.size * 16 for record in records)
    if expected_frames != gt.size:
        raise ValueError(f"frame alignment failed: prediction frames={expected_frames}, GT frames={gt.size}")

    rows: list[dict[str, object]] = []
    offset = 0
    for record in tqdm(records, desc="reliability quality against held-out GT", unit="video"):
        # Test CSV rows carry a feature-chunk suffix (``__0``), while the
        # shared hidden manifest is indexed by its video-level base key.
        key = base_key(record.source_path)
        if key not in test_hidden:
            raise FileNotFoundError(f"test hidden manifest has no artifact for {key}")
        hidden, _metadata = load_hidden_for_diagnostic(test_hidden[key])
        q, aligned, seeds, similarity = reliability_map(hidden, record.prob1, normal_mean, normal_std, config)
        expanded = bounded_top_indices(q, config.expand_top_p)
        frame_count = record.prob1.size * 16
        frame_gt = gt[offset:offset + frame_count].reshape(record.prob1.size, 16)
        offset += frame_count
        # The hidden cache may have a different temporal length.  The same
        # interpolation used at inference maps both the selected sets and q
        # to VadCLIP's score sequence before looking at held-out labels.
        q_scores = resample_to_score_length(q, record.prob1.size)
        seed_scores = top_indices(record.prob1, config.seed_top_p)
        expanded_scores = bounded_top_indices(q_scores, config.expand_top_p)
        seed_stat, expanded_stat = subset_stats(frame_gt, seed_scores), subset_stats(frame_gt, expanded_scores)
        row: dict[str, object] = {
            "key": key, "source_path": record.source_path, "label": record.label,
            "is_abnormal": not is_normal_label(args.dataset, record.label),
            "snippet_count": int(record.prob1.size), "frame_count": int(frame_count), "positive_frames": int(frame_gt.sum()),
            "hidden_length": int(hidden.shape[0]), "seed_hidden_count": int(seeds.size), "expanded_hidden_count": int(expanded.size),
            "reliability_mean": float(q_scores.mean()), "reliability_max": float(q_scores.max()),
            "score_mean": float(aligned.mean()), "seed_similarity_mean": float(similarity[seeds].mean()),
        }
        for prefix, values in (("seed", seed_stat), ("expanded", expanded_stat)):
            for name, value in values.items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
    if offset != gt.size:
        raise RuntimeError("internal GT offset did not consume all frame labels")
    write_rows(out_dir / "per_video_quality.csv", rows)
    seed_summary, expanded_summary = aggregate(rows, "seed"), aggregate(rows, "expanded")
    summary = {
        "method": "vadclip_seed_expand_reliability_quality_diagnostic_v1", "dataset": args.dataset,
        **config_as_dict(config), **score_audit,
        "test_videos": len(rows), "frame_labels_used_only_for_offline_diagnosis": True,
        "seed_top": seed_summary, "reliability_expanded_top": expanded_summary,
        "normal_reliability": aggregate_normal_reliability(rows),
        "decision_rule": "A formal M5 run is justified only if expanded precision remains high while expanded recall materially exceeds seed recall, and normal reliability stays low.",
    }
    save_json(out_dir / "summary.json", summary)
    print(
        f"seed: precision={seed_summary['micro_positive_rate']:.4f}, recall={seed_summary['micro_positive_recall']:.4f} | "
        f"expanded: precision={expanded_summary['micro_positive_rate']:.4f}, recall={expanded_summary['micro_positive_recall']:.4f}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'per_video_quality.csv'}", flush=True)


def load_hidden_for_diagnostic(path: str) -> tuple[np.ndarray, dict[str, object]]:
    """Local wrapper keeps imports at the module boundary easy to audit."""
    from common import load_hidden

    return load_hidden(path)


def resample_to_score_length(values: np.ndarray, target_length: int) -> np.ndarray:
    from common import resample_scores

    return resample_scores(values, target_length)


if __name__ == "__main__":
    main()
