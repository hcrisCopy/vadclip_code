#!/usr/bin/env python3
"""Measure how reliable VadCLIP top/bottom pseudo snippets are on labelled test data.

This is an offline analysis only.  Frame labels are read after frozen-baseline
inference to report pseudo-label quality; they never enter an optimiser,
checkpoint-selection rule, or a formal training CSV.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def add_injection_source() -> None:
    """Import shared utilities from this repository's neuron experiment only."""
    source = str(Path(__file__).resolve().parents[1] / "vadclip_neuron_injection")
    if source not in sys.path:
        sys.path.insert(0, source)


add_injection_source()
from common import clean_dir, ensure_dir, is_normal_label, load_clip_feature, read_csv, save_json  # noqa: E402


@dataclass(frozen=True)
class ScoreRecord:
    """One resumable frozen-baseline score sequence, in test-CSV order."""

    source_path: str
    label: str
    prob1: np.ndarray
    prob2: np.ndarray


def add_vadclip_source(vadclip_root: str) -> None:
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def baseline_options(dataset: str, vadclip_root: str):
    """Load the unmodified official option defaults for the requested dataset."""
    add_vadclip_source(vadclip_root)
    if dataset == "ucf":
        import ucf_option as option_module
    else:
        import xd_option as option_module
    return option_module.parser.parse_args([])


def state_dict_from_file(path: str) -> dict[str, torch.Tensor]:
    """Load a plain or checkpoint-wrapped baseline state dictionary."""
    try:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(path, map_location="cpu")
    state = artifact.get("model_state_dict", artifact) if isinstance(artifact, dict) else artifact
    if not isinstance(state, dict):
        raise ValueError(f"{path} does not contain a model state dictionary")
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def load_baseline(dataset: str, vadclip_root: str, model_path: str, device: torch.device):
    """Construct the official VadCLIP model without changing its code or weights."""
    options = baseline_options(dataset, vadclip_root)
    add_vadclip_source(vadclip_root)
    from model import CLIPVAD

    model = CLIPVAD(
        options.classes_num, options.embed_dim, options.visual_length, options.visual_width,
        options.visual_head, options.visual_layers, options.attn_window,
        options.prompt_prefix, options.prompt_postfix, str(device),
    )
    model.load_state_dict(state_dict_from_file(model_path), strict=True)
    model.to(device).eval()
    return model, options


def prompt_text(dataset: str) -> list[str]:
    """Use exactly the prompt order used by the existing pseudo-score stage."""
    if dataset == "ucf":
        from common import UCF_TEST_LABELS

        return list(UCF_TEST_LABELS.values())
    from common import XD_LABELS

    return list(XD_LABELS.values())


def score_feature(
    model,
    feature: np.ndarray,
    visual_length: int,
    prompts: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the existing pseudo-score chunking, then expose both VadCLIP heads.

    ``prob1`` is the classifier score actually used by global-768 selection.
    ``prob2`` is the language-branch anomaly probability used by XD AP2 and
    by the official detection-mAP path.
    """
    chunks, lengths = [], []
    for start in range(0, feature.shape[0], visual_length):
        part = feature[start:start + visual_length]
        if part.shape[0] == 0:
            continue
        lengths.append(part.shape[0])
        if part.shape[0] < visual_length:
            part = np.pad(part, ((0, visual_length - part.shape[0]), (0, 0)), mode="constant")
        chunks.append(part.reshape(1, visual_length, feature.shape[1]))
    if not chunks:
        raise ValueError("cannot score an empty feature sequence")
    visual = torch.from_numpy(np.concatenate(chunks, axis=0)).to(device)
    valid_lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    with torch.no_grad():
        _text, logits1, logits2 = model(visual, None, prompts, valid_lengths)
    original_length = feature.shape[0]
    probability1 = torch.sigmoid(logits1.reshape(-1)[:original_length])
    probability2 = 1.0 - logits2.reshape(-1, logits2.shape[-1])[:original_length].softmax(dim=-1)[:, 0]
    return (
        probability1.detach().cpu().numpy().astype(np.float32),
        probability2.detach().cpu().numpy().astype(np.float32),
    )


def artifact_path(score_dir: Path, source_path: str) -> Path:
    """Use the same one-file-per-test-video resume layout as test scripts."""
    return score_dir / f"{Path(source_path).stem}.npz"


def load_record(path: Path, source_path: str, label: str) -> ScoreRecord:
    """Validate an existing artifact before reusing it after interruption."""
    with np.load(path, allow_pickle=False) as artifact:
        required = {"prob1", "prob2", "source_path", "label"}
        if not required.issubset(artifact.files):
            raise ValueError(f"{path}: incomplete score artifact; rerun with --no-resume or --clean")
        saved_path = str(artifact["source_path"].item())
        saved_label = str(artifact["label"].item())
        if saved_path != source_path or saved_label != label:
            raise ValueError(f"{path}: does not match the current test CSV; rerun with --clean")
        prob1 = np.asarray(artifact["prob1"], dtype=np.float32).reshape(-1)
        prob2 = np.asarray(artifact["prob2"], dtype=np.float32).reshape(-1)
    if prob1.size == 0 or prob1.shape != prob2.shape or not np.isfinite(prob1).all() or not np.isfinite(prob2).all():
        raise ValueError(f"{path}: invalid stored baseline scores; rerun with --no-resume or --clean")
    return ScoreRecord(source_path, label, prob1, prob2)


def paired_indices(scores: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Exactly reproduce global-768's deterministic top/bottom pairing rule."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size < 2:
        raise ValueError(f"need at least two scores, got {values.size}")
    requested = max(1, int(np.ceil(float(top_p) * values.size)))
    count = min(requested, values.size // 2)
    order = np.argsort(values, kind="mergesort")
    return order[-count:][::-1].astype(np.int64), order[:count].astype(np.int64)


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a JSON-friendly optional ratio rather than fabricate a zero."""
    return float(numerator / denominator) if denominator else None


def optional_mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def subset_quality(frame_gt: np.ndarray, indices: np.ndarray) -> dict[str, int | float | None]:
    """Evaluate a selected snippet set against its aligned 16-frame GT blocks."""
    selected = frame_gt[indices]
    selected_frames = int(selected.size)
    selected_positive = int(selected.sum())
    video_positive = int(frame_gt.sum())
    return {
        "snippet_count": int(indices.size),
        "frame_count": selected_frames,
        "positive_frames": selected_positive,
        "positive_rate": safe_ratio(selected_positive, selected_frames),
        "positive_recall": safe_ratio(selected_positive, video_positive),
    }


def branch_video_quality(frame_gt: np.ndarray, scores: np.ndarray, top_p: float) -> dict[str, object]:
    """Return top-positive purity and bottom-negative contamination for one head."""
    top, bottom = paired_indices(scores, top_p)
    return {
        "top": subset_quality(frame_gt, top),
        "bottom": subset_quality(frame_gt, bottom),
        "top_indices": top,
        "bottom_indices": bottom,
    }


def overlap_quality(frame_gt: np.ndarray, left: np.ndarray, right: np.ndarray) -> dict[str, int | float | None]:
    """Measure agreement between the two heads' candidate sets."""
    intersection = np.intersect1d(left, right, assume_unique=False)
    union = np.union1d(left, right)
    details = subset_quality(frame_gt, intersection)
    details["jaccard"] = safe_ratio(len(intersection), len(union))
    return details


def flat_quality(prefix: str, quality: dict[str, object]) -> dict[str, object]:
    """Flatten a nested per-head result for a human-readable CSV row."""
    result: dict[str, object] = {}
    for side in ("top", "bottom"):
        for key, value in quality[side].items():
            result[f"{prefix}_{side}_{key}"] = value
    return result


def aggregate_branch(per_video: list[dict[str, object]], branch: str) -> dict[str, object]:
    """Aggregate micro and macro pseudo-label quality over abnormal videos only."""
    abnormal = [row for row in per_video if bool(row["is_abnormal"])]
    total_positive = sum(int(row["positive_frames"]) for row in abnormal)
    summary: dict[str, object] = {"abnormal_videos": len(abnormal), "positive_frames": total_positive}
    for side in ("top", "bottom"):
        frame_count = sum(int(row[f"{branch}_{side}_frame_count"]) for row in abnormal)
        selected_positive = sum(int(row[f"{branch}_{side}_positive_frames"]) for row in abnormal)
        precision_values = [row[f"{branch}_{side}_positive_rate"] for row in abnormal]
        recall_values = [row[f"{branch}_{side}_positive_recall"] for row in abnormal]
        summary[side] = {
            "selected_frames": frame_count,
            "selected_positive_frames": selected_positive,
            "micro_positive_rate": safe_ratio(selected_positive, frame_count),
            "micro_positive_recall": safe_ratio(selected_positive, total_positive),
            "macro_positive_rate": optional_mean(precision_values),
            "macro_positive_recall": optional_mean(recall_values),
        }
    return summary


def aggregate_overlap(per_video: list[dict[str, object]], side: str) -> dict[str, object]:
    """Aggregate classifier/language candidate-set overlap on abnormal videos."""
    abnormal = [row for row in per_video if bool(row["is_abnormal"])]
    frame_count = sum(int(row[f"agreement_{side}_frame_count"]) for row in abnormal)
    selected_positive = sum(int(row[f"agreement_{side}_positive_frames"]) for row in abnormal)
    total_positive = sum(int(row["positive_frames"]) for row in abnormal)
    return {
        "micro_positive_rate": safe_ratio(selected_positive, frame_count),
        "micro_positive_recall": safe_ratio(selected_positive, total_positive),
        "macro_positive_rate": optional_mean([row[f"agreement_{side}_positive_rate"] for row in abnormal]),
        "macro_positive_recall": optional_mean([row[f"agreement_{side}_positive_recall"] for row in abnormal]),
        "macro_jaccard": optional_mean([row[f"agreement_{side}_jaccard"] for row in abnormal]),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a stable CSV even when individual videos have no positive frames."""
    if not rows:
        raise RuntimeError("no per-video diagnostic rows were produced")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_branch_summary(name: str, summary: dict[str, object]) -> None:
    """Print the four quantities that answer the pseudo-label reliability question."""
    top, bottom = summary["top"], summary["bottom"]
    print(
        f"{name}: top precision={top['micro_positive_rate']:.4f}, "
        f"top recall={top['micro_positive_recall']:.4f}, "
        f"bottom contamination={bottom['micro_positive_rate']:.4f}, "
        f"bottom missed-positive recall={bottom['micro_positive_recall']:.4f}",
        flush=True,
    )


def format_metric(value: float | None) -> str:
    """Keep the terminal report usable even for a degenerate label split."""
    return f"{value:.4f}" if value is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline GT diagnostic for VadCLIP top/bottom pseudo snippets.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--test-list", required=True, help="Official 512D CLIP test CSV; never a training input.")
    parser.add_argument("--model-path", required=True, help="Frozen official VadCLIP baseline checkpoint.")
    parser.add_argument("--gt-path", required=True, help="Official frame GT used only after inference for analysis.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vadclip-root", default="VadCLIP")
    parser.add_argument("--top-p", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean", action="store_true", help="Remove only this diagnostic output directory first.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute per-video scores but preserve the output directory.")
    args = parser.parse_args()

    if not 0.0 < args.top_p <= 0.5:
        parser.error("--top-p must be in (0, 0.5] so top and bottom sets remain disjoint")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    for path in (args.test_list, args.model_path, args.gt_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"missing diagnostic input: {path}")

    device = torch.device(args.device)
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    score_dir = ensure_dir(out_dir / "per_video_scores")
    test_frame = read_csv(args.test_list).reset_index(drop=True)
    gt = np.asarray(np.load(args.gt_path), dtype=np.int8).reshape(-1)
    if not np.isin(gt, [0, 1]).all():
        raise ValueError(f"{args.gt_path}: expected binary frame labels")
    save_json(out_dir / "run_config.json", {
        "method": "vadclip_pseudo_label_quality_diagnostic_v1",
        "dataset": args.dataset,
        "test_list": args.test_list,
        "model_path": args.model_path,
        "gt_path": args.gt_path,
        "top_p": float(args.top_p),
        "vadclip_root": args.vadclip_root,
        "frame_labels_used_only_for_offline_diagnosis": True,
        "model_parameters_updated": False,
        "formal_training_or_checkpoint_selection_changed": False,
    })

    model, options = load_baseline(args.dataset, args.vadclip_root, args.model_path, device)
    prompts = prompt_text(args.dataset)
    records: list[ScoreRecord] = []
    used_outputs: set[Path] = set()
    for _, row in tqdm(test_frame.iterrows(), total=len(test_frame), desc="frozen baseline pseudo scores", unit="video"):
        source_path, label = str(row["path"]), str(row["label"])
        output_path = artifact_path(score_dir, source_path)
        if output_path in used_outputs:
            raise ValueError(f"duplicate test feature stem creates ambiguous resume artifact: {output_path}")
        used_outputs.add(output_path)
        if output_path.is_file() and not args.no_resume:
            record = load_record(output_path, source_path, label)
        else:
            feature = load_clip_feature(source_path)
            if feature.shape[1] != options.visual_width:
                raise ValueError(
                    f"{source_path}: expected original {options.visual_width}D CLIP feature, got {feature.shape[1]}D; "
                    "use the official 512D test CSV, not a 1280D concat CSV"
                )
            prob1, prob2 = score_feature(model, feature, options.visual_length, prompts, device)
            np.savez_compressed(
                output_path, prob1=prob1, prob2=prob2,
                source_path=np.asarray(source_path), label=np.asarray(label),
            )
            record = ScoreRecord(source_path, label, prob1, prob2)
        records.append(record)

    expected_frames = sum(record.prob1.size * 16 for record in records)
    if expected_frames != gt.size:
        raise ValueError(
            f"frame alignment failed: scores imply {expected_frames} frames but {args.gt_path} has {gt.size}; "
            "the test CSV/GT pair must follow the official dataset order"
        )

    rows: list[dict[str, object]] = []
    offset = 0
    for record in tqdm(records, desc="GT pseudo-label quality", unit="video"):
        count = int(record.prob1.size)
        frame_gt = gt[offset:offset + count * 16].reshape(count, 16)
        offset += count * 16
        classifier = branch_video_quality(frame_gt, record.prob1, args.top_p)
        language = branch_video_quality(frame_gt, record.prob2, args.top_p)
        top_agreement = overlap_quality(frame_gt, classifier["top_indices"], language["top_indices"])
        bottom_agreement = overlap_quality(frame_gt, classifier["bottom_indices"], language["bottom_indices"])
        correlation = None
        if count > 1 and np.std(record.prob1) > 0.0 and np.std(record.prob2) > 0.0:
            correlation = float(np.corrcoef(record.prob1, record.prob2)[0, 1])
        row: dict[str, object] = {
            "key": Path(record.source_path).stem,
            "source_path": record.source_path,
            "label": record.label,
            "is_abnormal": not is_normal_label(args.dataset, record.label),
            "snippet_count": count,
            "frame_count": int(frame_gt.size),
            "positive_frames": int(frame_gt.sum()),
            "classifier_language_score_correlation": correlation,
        }
        row.update(flat_quality("classifier", classifier))
        row.update(flat_quality("language", language))
        for side, agreement in (("top", top_agreement), ("bottom", bottom_agreement)):
            for key, value in agreement.items():
                row[f"agreement_{side}_{key}"] = value
        rows.append(row)
    if offset != gt.size:
        raise RuntimeError("internal frame-GT offset did not consume the expected label array")

    write_rows(out_dir / "per_video_quality.csv", rows)
    classifier_summary = aggregate_branch(rows, "classifier")
    language_summary = aggregate_branch(rows, "language")
    abnormal_correlations = [
        row["classifier_language_score_correlation"] for row in rows if bool(row["is_abnormal"])
    ]
    summary = {
        "method": "vadclip_pseudo_label_quality_diagnostic_v1",
        "dataset": args.dataset,
        "top_p": float(args.top_p),
        "test_videos": len(rows),
        "abnormal_test_videos": int(sum(bool(row["is_abnormal"]) for row in rows)),
        "frame_labels_used_only_for_offline_diagnosis": True,
        "current_global768_selection_head": "classifier_prob1_sigmoid_logits1",
        "classifier_prob1": classifier_summary,
        "language_prob2": language_summary,
        "head_agreement": {
            "top": aggregate_overlap(rows, "top"),
            "bottom": aggregate_overlap(rows, "bottom"),
            "macro_score_correlation": optional_mean(abnormal_correlations),
        },
        "interpretation": {
            "top_precision": "Higher means pseudo-positive snippets contain more true anomalous frames.",
            "top_recall": "Higher means pseudo-positive snippets cover more true anomalous frames.",
            "bottom_contamination": "Lower means pseudo-negative snippets contain fewer true anomalous frames.",
            "bottom_missed_positive_recall": "Lower means fewer true anomalies were incorrectly placed in the pseudo-negative set.",
        },
    }
    save_json(out_dir / "summary.json", summary)
    print_branch_summary("classifier logits1 (current global-768 selector)", classifier_summary)
    print_branch_summary("language logits2 (official XD AP2/mAP branch)", language_summary)
    agreement = summary["head_agreement"]
    print(
        f"head agreement: top Jaccard={format_metric(agreement['top']['macro_jaccard'])}, "
        f"bottom Jaccard={format_metric(agreement['bottom']['macro_jaccard'])}, "
        f"score correlation={format_metric(agreement['macro_score_correlation'])}",
        flush=True,
    )
    print(f"wrote {out_dir / 'summary.json'} and {out_dir / 'per_video_quality.csv'}", flush=True)


if __name__ == "__main__":
    main()
