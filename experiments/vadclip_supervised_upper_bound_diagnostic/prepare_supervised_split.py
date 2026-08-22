#!/usr/bin/env python3
"""Create a disjoint video split with GT files for a supervised diagnostic."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared import add_injection_source

add_injection_source()
from common import base_key, clean_dir, ensure_dir, is_normal_label, load_clip_feature, read_csv, save_json, write_csv


def label_tokens(dataset: str, label: str) -> tuple[str, ...]:
    """Return official event codes to report stratification coverage."""
    if dataset == "ucf":
        return (str(label),)
    tokens = tuple(token for token in str(label).split("-") if token and token != "0")
    return tokens or (str(label),)


def make_groups(frame: pd.DataFrame) -> list[tuple[str, str, tuple[int, ...]]]:
    """Group CSV records into unsplittable original videos."""
    grouped: dict[str, list[int]] = {}
    labels: dict[str, str] = {}
    for index, row in frame.iterrows():
        key, label = base_key(str(row["path"])), str(row["label"])
        if key in labels and labels[key] != label:
            raise ValueError(f"test CSV video {key!r} has inconsistent labels")
        labels[key] = label
        grouped.setdefault(key, []).append(int(index))
    return [(key, labels[key], tuple(indices)) for key, indices in grouped.items()]


def split_counts(count: int, train_fraction: float, validation_fraction: float) -> tuple[int, int, int]:
    """Allocate a stratum while preserving a held-out video when possible."""
    train = max(1, int(np.floor(train_fraction * count)))
    validation = max(1, int(np.floor(validation_fraction * count))) if count >= 3 else 0
    heldout = count - train - validation
    if heldout <= 0 and count >= 2:
        heldout, train = 1, max(1, train - 1)
    if train + validation + heldout != count:
        raise RuntimeError("invalid stratum allocation")
    return train, validation, heldout


def stratified_split(
    groups: list[tuple[str, str, tuple[int, ...]]], train_fraction: float, validation_fraction: float, seed: int
) -> dict[str, list[tuple[str, str, tuple[int, ...]]]]:
    """Split complete videos by their official labels, deterministically."""
    by_label: dict[str, list[tuple[str, str, tuple[int, ...]]]] = defaultdict(list)
    for group in groups:
        by_label[group[1]].append(group)
    rng = np.random.default_rng(seed)
    result = {"train": [], "validation": [], "heldout": []}
    for label in sorted(by_label):
        values = by_label[label]
        values = [values[index] for index in rng.permutation(len(values))]
        train_count, validation_count, heldout_count = split_counts(len(values), train_fraction, validation_fraction)
        result["train"].extend(values[:train_count])
        result["validation"].extend(values[train_count:train_count + validation_count])
        result["heldout"].extend(values[train_count + validation_count:train_count + validation_count + heldout_count])
    all_keys = [group[0] for groups_for_split in result.values() for group in groups_for_split]
    if len(all_keys) != len(groups) or len(set(all_keys)) != len(groups):
        raise RuntimeError("video split is not a disjoint partition")
    return result


def object_subset(values: np.ndarray, indices: list[int]) -> np.ndarray:
    """Subset object annotations without NumPy collapsing matching shapes."""
    result = np.empty(len(indices), dtype=object)
    for output_index, source_index in enumerate(indices):
        result[output_index] = values[source_index]
    return result


def coverage(dataset: str, groups: list[tuple[str, str, tuple[int, ...]]]) -> dict[str, int]:
    values: dict[str, int] = defaultdict(int)
    for _key, label, _rows in groups:
        for token in label_tokens(dataset, label):
            values[token] += 1
    return dict(sorted(values.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create disjoint test-video splits for a supervised upper-bound diagnostic.")
    parser.add_argument("--dataset", choices=["ucf", "xd"], required=True)
    parser.add_argument("--full-test-list", required=True)
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
        parser.error("train-fraction + validation-fraction must leave a held-out fraction")
    out_dir = clean_dir(args.out_dir) if args.clean else ensure_dir(args.out_dir)
    manifest_path = out_dir / "split_manifest.json"
    if manifest_path.exists() and not args.no_resume:
        print(f"reuse completed supervised diagnostic split: {manifest_path}", flush=True)
        return

    frame = read_csv(args.full_test_list).reset_index(drop=True)
    gt = np.asarray(np.load(args.full_gt_path), dtype=np.float32).reshape(-1)
    segments = np.load(args.full_segment_path, allow_pickle=True)
    labels = np.load(args.full_label_path, allow_pickle=True)
    if len(segments) != len(frame) or len(labels) != len(frame):
        raise ValueError("official segment/label arrays must have one entry per test CSV record")
    frame_offsets, alignment_rows, offset = [], [], 0
    for index, row in tqdm(frame.iterrows(), total=len(frame), desc="verify frame alignment", unit="record"):
        path = str(row["path"])
        snippets = int(load_clip_feature(path).shape[0])
        frame_count = snippets * 16
        frame_offsets.append((offset, offset + frame_count))
        alignment_rows.append([index, base_key(path), path, str(row["label"]), snippets, offset, offset + frame_count])
        offset += frame_count
    if offset != len(gt):
        raise ValueError(f"full test frame alignment failed: features={offset}, official GT={len(gt)}")

    groups = make_groups(frame)
    split = stratified_split(groups, args.train_fraction, args.validation_fraction, args.seed)
    annotation_dir = ensure_dir(out_dir / "annotations")
    manifest_splits: dict[str, dict[str, object]] = {}
    split_rows = []
    for name, groups_for_split in split.items():
        normal = sum(is_normal_label(args.dataset, group[1]) for group in groups_for_split)
        if normal == 0 or normal == len(groups_for_split):
            raise RuntimeError(f"{name} must contain normal and abnormal videos")
        indices = sorted(index for _key, _label, rows in groups_for_split for index in rows)
        subset = frame.iloc[indices].reset_index(drop=True)
        subset.to_csv(out_dir / f"{name}.csv", index=False)
        subset_gt = np.concatenate([gt[left:right] for left, right in (frame_offsets[index] for index in indices)])
        gt_path = annotation_dir / f"{name}_gt.npy"
        segment_path = annotation_dir / f"{name}_segment.npy"
        label_path = annotation_dir / f"{name}_label.npy"
        np.save(gt_path, subset_gt)
        np.save(segment_path, object_subset(segments, indices), allow_pickle=True)
        np.save(label_path, object_subset(labels, indices), allow_pickle=True)
        manifest_splits[name] = {
            "videos": len(groups_for_split), "records": len(indices), "frame_count": int(len(subset_gt)),
            "label_coverage": coverage(args.dataset, groups_for_split), "csv": str(out_dir / f"{name}.csv"),
            "gt_path": str(gt_path), "segment_path": str(segment_path), "label_path": str(label_path),
        }
        split_rows.extend([key, label, name, len(rows)] for key, label, rows in groups_for_split)
    write_csv(out_dir / "video_split.csv", ["video_key", "label", "split", "records"], split_rows)
    write_csv(
        out_dir / "record_alignment.csv",
        ["record_index", "video_key", "path", "label", "snippets", "frame_start", "frame_end"], alignment_rows,
    )
    save_json(manifest_path, {
        "method": "supervised_global768_upper_bound_diagnostic_v1",
        "dataset": args.dataset,
        "description": "A disjoint test-video partition. Only train_gt.npy may supervise the diagnostic adapter; validation selects a checkpoint and heldout remains untouched.",
        "source": {"full_test_list": args.full_test_list, "full_gt_path": args.full_gt_path, "full_segment_path": args.full_segment_path, "full_label_path": args.full_label_path},
        "fractions": {"train": args.train_fraction, "validation": args.validation_fraction, "heldout": 1.0 - args.train_fraction - args.validation_fraction},
        "seed": args.seed, "total_videos": len(groups), "total_records": len(frame), "total_frames": int(len(gt)),
        "splits": manifest_splits, "training_uses_frame_labels": True,
    })
    print("created disjoint supervised split | " + " | ".join(
        f"{name}: videos={info['videos']} records={info['records']} frames={info['frame_count']}"
        for name, info in manifest_splits.items()
    ), flush=True)


if __name__ == "__main__":
    main()
