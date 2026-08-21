#!/usr/bin/env python3
"""Create leakage-free video splits and aligned GT subsets for diagnosis.

This script never creates training targets from frame labels.  It merely splits
the already labelled test set by video, then copies the corresponding GT,
segment and class-label entries so the official VadCLIP metrics can be run on
the validation and held-out subsets without frame-order drift.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared import add_injection_source

add_injection_source()
from common import base_key, clean_dir, ensure_dir, is_normal_label, load_clip_feature, read_csv, save_json, write_csv


@dataclass(frozen=True)
class VideoGroup:
    key: str
    label: str
    row_indices: tuple[int, ...]


def label_tokens(dataset: str, label: str) -> tuple[str, ...]:
    """Return the official video-level event codes for coverage reporting."""
    if dataset == "ucf":
        return (str(label),)
    tokens = tuple(token for token in str(label).split("-") if token and token != "0")
    return tokens or (str(label),)


def make_video_groups(frame: pd.DataFrame) -> list[VideoGroup]:
    grouped: dict[str, list[int]] = {}
    labels: dict[str, str] = {}
    for index, row in frame.iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"test CSV video {key!r} has inconsistent labels {labels[key]!r} and {label!r}")
        labels[key] = label
        grouped.setdefault(key, []).append(int(index))
    return [VideoGroup(key, labels[key], tuple(indices)) for key, indices in grouped.items()]


def split_counts(count: int, train_fraction: float, validation_fraction: float) -> tuple[int, int, int]:
    """Allocate each label stratum while retaining a held-out set when possible."""
    if count <= 0:
        raise ValueError("cannot split an empty stratum")
    train = max(1, int(np.floor(train_fraction * count)))
    validation = max(1, int(np.floor(validation_fraction * count))) if count >= 3 else 0
    heldout = count - train - validation
    if heldout <= 0 and count >= 2:
        heldout, train = 1, max(1, train - 1)
    if train + validation + heldout != count:
        raise RuntimeError("invalid stratum allocation")
    return train, validation, heldout


def stratified_video_split(
    groups: list[VideoGroup], train_fraction: float, validation_fraction: float, seed: int
) -> dict[str, list[VideoGroup]]:
    """Split whole videos by their exact official video label deterministically."""
    by_label: dict[str, list[VideoGroup]] = defaultdict(list)
    for group in groups:
        by_label[group.label].append(group)
    rng = np.random.default_rng(seed)
    split = {"train": [], "validation": [], "heldout": []}
    for label in sorted(by_label):
        stratum = by_label[label]
        order = rng.permutation(len(stratum))
        shuffled = [stratum[index] for index in order]
        train_count, validation_count, heldout_count = split_counts(
            len(shuffled), train_fraction, validation_fraction
        )
        split["train"].extend(shuffled[:train_count])
        split["validation"].extend(shuffled[train_count:train_count + validation_count])
        split["heldout"].extend(shuffled[train_count + validation_count:train_count + validation_count + heldout_count])
    all_keys = [group.key for groups_for_split in split.values() for group in groups_for_split]
    if len(all_keys) != len(groups) or len(set(all_keys)) != len(groups):
        raise RuntimeError("video split is not a disjoint partition")
    return split


def ordered_rows(groups: list[VideoGroup]) -> list[int]:
    return sorted(index for group in groups for index in group.row_indices)


def object_subset(values: np.ndarray, indices: list[int]) -> np.ndarray:
    """Keep an object array one-dimensional even if individual entries match."""
    result = np.empty(len(indices), dtype=object)
    for output_index, source_index in enumerate(indices):
        result[output_index] = values[source_index]
    return result


def coverage(dataset: str, groups: list[VideoGroup]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for group in groups:
        for token in label_tokens(dataset, group.label):
            counts[token] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-level test split with aligned official VadCLIP annotations.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--full-test-list", required=True, help="Full global-768 concat test CSV in official evaluation order.")
    parser.add_argument("--full-gt-path", required=True)
    parser.add_argument("--full-segment-path", required=True)
    parser.add_argument("--full-label-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0 or not 0.0 < args.validation_fraction < 1.0:
        parser.error("train-fraction and validation-fraction must be in (0, 1)")
    if args.train_fraction + args.validation_fraction >= 1.0:
        parser.error("train-fraction + validation-fraction must leave a positive held-out fraction")
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    manifest_path = out_dir / "split_manifest.json"
    if manifest_path.exists() and not args.no_resume:
        print(f"reuse completed video split: {manifest_path}", flush=True)
        return

    frame = read_csv(args.full_test_list).reset_index(drop=True)
    groups = make_video_groups(frame)
    gt = np.asarray(np.load(args.full_gt_path)).reshape(-1)
    segments = np.load(args.full_segment_path, allow_pickle=True)
    labels = np.load(args.full_label_path, allow_pickle=True)
    if len(segments) != len(frame) or len(labels) != len(frame):
        raise ValueError(
            "official segment/label arrays must have one entry per full test CSV row: "
            f"csv={len(frame)}, segments={len(segments)}, labels={len(labels)}"
        )

    frame_offsets, offset, alignment_rows = [], 0, []
    for index, row in tqdm(frame.iterrows(), total=len(frame), desc="verify test-frame alignment", unit="video"):
        path = str(row["path"])
        snippets = int(load_clip_feature(path).shape[0])
        frame_count = snippets * 16
        frame_offsets.append((offset, offset + frame_count))
        alignment_rows.append([index, base_key(path), path, str(row["label"]), snippets, offset, offset + frame_count])
        offset += frame_count
    if offset != len(gt):
        raise ValueError(
            f"full test frame alignment failed: concat features imply {offset} frames, official GT has {len(gt)}"
        )

    split = stratified_video_split(groups, args.train_fraction, args.validation_fraction, args.seed)
    for name, groups_for_split in split.items():
        normal = sum(is_normal_label(args.dataset, group.label) for group in groups_for_split)
        abnormal = len(groups_for_split) - normal
        if normal == 0 or abnormal == 0:
            raise RuntimeError(
                f"{name} split needs both normal and abnormal videos; got normal={normal}, abnormal={abnormal}. "
                "Choose a different seed or larger split fractions."
            )
    annotation_dir = ensure_dir(out_dir / "annotations")
    split_rows = []
    manifest_splits: dict[str, dict[str, object]] = {}
    for name, split_groups in split.items():
        indices = ordered_rows(split_groups)
        if not indices:
            raise RuntimeError(f"{name} split is empty; choose different fractions or seed")
        split_frame = frame.iloc[indices].reset_index(drop=True)
        split_csv = out_dir / f"{name}.csv"
        split_frame.to_csv(split_csv, index=False)
        split_rows.extend([group.key, group.label, name, len(group.row_indices)] for group in split_groups)
        manifest_splits[name] = {
            "videos": len(split_groups),
            "records": len(indices),
            "frame_count": int(sum(frame_offsets[index][1] - frame_offsets[index][0] for index in indices)),
            "label_coverage": coverage(args.dataset, split_groups),
            "csv": str(split_csv),
        }
        if name == "train":
            continue
        gt_subset = np.concatenate([gt[left:right] for left, right in (frame_offsets[index] for index in indices)])
        gt_path = annotation_dir / f"{name}_gt.npy"
        segment_path = annotation_dir / f"{name}_segment.npy"
        label_path = annotation_dir / f"{name}_label.npy"
        np.save(gt_path, gt_subset)
        np.save(segment_path, object_subset(segments, indices), allow_pickle=True)
        np.save(label_path, object_subset(labels, indices), allow_pickle=True)
        manifest_splits[name].update({
            "gt_path": str(gt_path), "segment_path": str(segment_path), "label_path": str(label_path),
        })
    write_csv(out_dir / "video_split.csv", ["video_key", "label", "split", "records"], split_rows)
    write_csv(
        out_dir / "record_alignment.csv",
        ["record_index", "video_key", "path", "label", "snippets", "frame_start", "frame_end"],
        alignment_rows,
    )
    save_json(manifest_path, {
        "method": "video_level_original_method_diagnostic_v1",
        "dataset": args.dataset,
        "description": "Disjoint test-video partition for a no-trick original-method diagnostic. Frame labels are written only for validation/held-out evaluation, never for training.",
        "source": {
            "full_test_list": args.full_test_list,
            "full_gt_path": args.full_gt_path,
            "full_segment_path": args.full_segment_path,
            "full_label_path": args.full_label_path,
        },
        "fractions": {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "heldout": 1.0 - args.train_fraction - args.validation_fraction,
        },
        "seed": args.seed,
        "total_videos": len(groups),
        "total_records": len(frame),
        "total_frames": int(len(gt)),
        "splits": manifest_splits,
        "training_uses_frame_labels": False,
    })
    print(
        "created disjoint video split | " + " | ".join(
            f"{name}: videos={info['videos']} records={info['records']} frames={info['frame_count']}"
            for name, info in manifest_splits.items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
