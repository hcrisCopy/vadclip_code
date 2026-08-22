#!/usr/bin/env python3
"""GT-isolated diagnostic for normal-reference novelty reliability.

The held-out frame labels are read only after all label-free q values have
been generated.  They cannot affect calibration, neuron selection, training,
or checkpoint selection.
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
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


add_sources()
from analyze_pseudo_label_quality import (  # noqa: E402
    ScoreRecord,
    artifact_path,
    load_baseline,
    load_record,
    prompt_text,
    score_feature,
)
from common import base_key, clean_dir, ensure_dir, is_normal_label, load_clip_feature, load_hidden, read_csv, resample_scores, save_json  # noqa: E402
from normal_novelty import (  # noqa: E402
    calibration_dict,
    expanded_indices,
    load_or_build_calibration,
    manifest_map,
    reliability_map,
    top_indices,
)


def subset_stats(frame_gt: np.ndarray, indices: np.ndarray) -> dict[str, int | float | None]:
    selected = frame_gt[indices]
    positives, total, video_positives = int(selected.sum()), int(selected.size), int(frame_gt.sum())
    return {
        "snippet_count": int(indices.size), "frame_count": total, "positive_frames": positives,
        "positive_rate": float(positives / total) if total else None,
        "positive_recall": float(positives / video_positives) if video_positives else None,
    }


def aggregate(rows: list[dict[str, object]], prefix: str) -> dict[str, float | int | None]:
    abnormal = [row for row in rows if bool(row["is_abnormal"])]
    selected_frames = sum(int(row[f"{prefix}_frame_count"]) for row in abnormal)
    selected_positive = sum(int(row[f"{prefix}_positive_frames"]) for row in abnormal)
    all_positive = sum(int(row["positive_frames"]) for row in abnormal)
    macro_precision = [row[f"{prefix}_positive_rate"] for row in abnormal if row[f"{prefix}_positive_rate"] is not None]
    macro_recall = [row[f"{prefix}_positive_recall"] for row in abnormal if row[f"{prefix}_positive_recall"] is not None]
    return {
        "abnormal_videos": len(abnormal), "selected_frames": selected_frames, "selected_positive_frames": selected_positive,
        "micro_positive_rate": float(selected_positive / selected_frames) if selected_frames else None,
        "micro_positive_recall": float(selected_positive / all_positive) if all_positive else None,
        "macro_positive_rate": float(np.mean(macro_precision)) if macro_precision else None,
        "macro_positive_recall": float(np.mean(macro_recall)) if macro_recall else None,
    }


def normal_q_summary(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    normal = [row for row in rows if not bool(row["is_abnormal"])]
    maxima, means = [float(row["q_max"] ) for row in normal], [float(row["q_mean"]) for row in normal]
    return {
        "normal_videos": len(normal), "mean_of_video_means": float(np.mean(means)) if means else None,
        "mean_of_video_maxima": float(np.mean(maxima)) if maxima else None,
        "p95_video_maximum": float(np.quantile(maxima, 0.95)) if maxima else None,
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline XD frame-label diagnostic for normal-novelty reliability q.")
    parser.add_argument("--dataset", choices=["xd", "ucf"], required=True)
    parser.add_argument("--source-train-csv", required=True)
    parser.add_argument("--train-hidden-manifest", required=True)
    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--test-hidden-manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--seed-top-p", type=float, default=0.10)
    parser.add_argument("--expand-top-p", type=float, default=0.30)
    parser.add_argument("--normal-score-quantile", type=float, default=0.95)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--normal-novelty-quantile", type=float, default=0.95)
    parser.add_argument("--novelty-temperature-scale", type=float, default=1.0)
    parser.add_argument("--normal-score-snippets-per-video", type=int, default=256)
    parser.add_argument("--normal-hidden-snippets-per-video", type=int, default=256)
    parser.add_argument("--sigma-min", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if args.clean and args.no_resume:
        parser.error("--clean and --no-resume cannot be used together")
    for value in (args.source_train_csv, args.train_hidden_manifest, args.pseudo_csv, args.test_list, args.test_hidden_manifest, args.model_path, args.gt_path):
        if not Path(value).is_file():
            raise FileNotFoundError(f"missing diagnostic input: {value}")
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    calibration, mean, std, audit = load_or_build_calibration(
        out_dir, args.dataset, args.source_train_csv, args.train_hidden_manifest, args.pseudo_csv,
        args.normal_score_quantile, args.normal_novelty_quantile, args.score_temperature,
        args.novelty_temperature_scale, args.seed_top_p, args.expand_top_p,
        args.normal_score_snippets_per_video, args.normal_hidden_snippets_per_video, args.sigma_min, args.no_resume,
    )
    test_hidden = manifest_map(args.test_hidden_manifest)
    score_dir = ensure_dir(out_dir / "per_video_scores")
    test_frame = read_csv(args.test_list).reset_index(drop=True)
    gt = np.asarray(np.load(args.gt_path), dtype=np.int8).reshape(-1)
    if not np.isin(gt, [0, 1]).all():
        raise ValueError("frame GT must contain only 0/1")
    save_json(out_dir / "run_config.json", {
        "method": "vadclip_normal_novelty_reliability_diagnostic_v1", "dataset": args.dataset,
        "source_train_csv": args.source_train_csv, "test_list": args.test_list, "gt_path": args.gt_path,
        **calibration_dict(calibration), **audit,
        "frame_labels_used_only_for_offline_diagnosis": True,
        "formal_training_or_checkpoint_selection_changed": False,
    })

    device = torch.device(args.device)
    model, options = load_baseline(args.dataset, args.vadclip_root, args.model_path, device)
    prompts = prompt_text(args.dataset)
    records: list[ScoreRecord] = []
    occupied: set[Path] = set()
    for _, row in tqdm(test_frame.iterrows(), total=len(test_frame), desc="frozen baseline scores", unit="video"):
        source_path, label = str(row["path"]), str(row["label"])
        output = artifact_path(score_dir, source_path)
        if output in occupied:
            raise ValueError(f"duplicate test feature stem: {output}")
        occupied.add(output)
        if output.is_file() and not args.no_resume:
            record = load_record(output, source_path, label)
        else:
            feature = load_clip_feature(source_path)
            if feature.shape[1] != options.visual_width:
                raise ValueError(f"{source_path}: expected {options.visual_width}D official feature")
            prob1, prob2 = score_feature(model, feature, options.visual_length, prompts, device)
            np.savez_compressed(output, prob1=prob1, prob2=prob2, source_path=np.asarray(source_path), label=np.asarray(label))
            record = ScoreRecord(source_path, label, prob1, prob2)
        records.append(record)
    if sum(record.prob1.size * 16 for record in records) != gt.size:
        raise ValueError("test CSV and frame GT do not align")

    rows, offset = [], 0
    for record in tqdm(records, desc="normal-novelty quality against held-out GT", unit="video"):
        key = base_key(record.source_path)
        if key not in test_hidden:
            raise FileNotFoundError(f"test hidden manifest has no artifact for {key}")
        hidden, _metadata = load_hidden(test_hidden[key])
        q_hidden, _scores, novelty = reliability_map(hidden, record.prob1, mean, std, calibration)
        q = resample_scores(q_hidden, record.prob1.size)
        frame_gt = gt[offset:offset + record.prob1.size * 16].reshape(record.prob1.size, 16)
        offset += record.prob1.size * 16
        seed_stat = subset_stats(frame_gt, top_indices(record.prob1, calibration.seed_top_p))
        expanded_stat = subset_stats(frame_gt, expanded_indices(q, calibration.expand_top_p))
        row: dict[str, object] = {
            "key": key, "source_path": record.source_path, "label": record.label,
            "is_abnormal": not is_normal_label(args.dataset, record.label), "snippet_count": int(record.prob1.size),
            "frame_count": int(frame_gt.size), "positive_frames": int(frame_gt.sum()), "hidden_length": int(hidden.shape[0]),
            "q_mean": float(q.mean()), "q_max": float(q.max()), "novelty_mean": float(novelty.mean()), "novelty_max": float(novelty.max()),
        }
        for prefix, values in (("seed", seed_stat), ("expanded", expanded_stat)):
            for name, value in values.items():
                row[f"{prefix}_{name}"] = value
        rows.append(row)
    if offset != gt.size:
        raise RuntimeError("failed to consume all frame labels")
    write_rows(out_dir / "per_video_quality.csv", rows)
    seed, expanded = aggregate(rows, "seed"), aggregate(rows, "expanded")
    summary = {
        "method": "vadclip_normal_novelty_reliability_diagnostic_v1", "dataset": args.dataset,
        **calibration_dict(calibration), **audit, "test_videos": len(rows),
        "frame_labels_used_only_for_offline_diagnosis": True,
        "seed_top": seed, "normal_novelty_expanded_top": expanded, "normal_reliability": normal_q_summary(rows),
        "pass_criteria": "Expanded precision should remain near seed precision, recall should substantially exceed seed recall, and normal q p95 maximum should be low.",
    }
    save_json(out_dir / "summary.json", summary)
    print(
        f"seed: precision={seed['micro_positive_rate']:.4f}, recall={seed['micro_positive_recall']:.4f} | "
        f"normal-novelty expanded: precision={expanded['micro_positive_rate']:.4f}, recall={expanded['micro_positive_recall']:.4f}",
        flush=True,
    )
    print(f"normal q p95 video maximum={summary['normal_reliability']['p95_video_maximum']:.4f}", flush=True)
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'per_video_quality.csv'}", flush=True)


if __name__ == "__main__":
    main()
