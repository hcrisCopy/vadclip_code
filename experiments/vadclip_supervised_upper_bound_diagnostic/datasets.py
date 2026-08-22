"""Frame-label aligned concat-feature dataset for the diagnostic split only."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from shared import add_injection_source

add_injection_source()
from common import load_clip_feature, read_csv


def process_feature_and_target(
    feature: np.ndarray, target: np.ndarray, visual_length: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Apply VadCLIP's training reduction to features and binary targets.

    A reduced target bin is positive when any of its original 16-frame
    snippets is positive.  Padding targets are zero and are later excluded by
    the returned valid length.
    """
    feature = np.asarray(feature, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    original_length = int(feature.shape[0])
    if target.size != original_length:
        raise ValueError(f"feature/target lengths disagree: features={original_length}, target={target.size}")
    if original_length > visual_length:
        reduced_feature = np.zeros((visual_length, feature.shape[1]), dtype=np.float32)
        reduced_target = np.zeros(visual_length, dtype=np.float32)
        boundaries = np.linspace(0, original_length, visual_length + 1, dtype=np.int32)
        for index in range(visual_length):
            left, right = int(boundaries[index]), int(boundaries[index + 1])
            reduced_feature[index] = feature[left:right].mean(axis=0) if left != right else feature[left]
            reduced_target[index] = target[left:right].max() if left != right else target[left]
        return reduced_feature, reduced_target, visual_length
    if original_length < visual_length:
        feature = np.pad(feature, ((0, visual_length - original_length), (0, 0)), mode="constant")
        target = np.pad(target, (0, visual_length - original_length), mode="constant")
    return feature.astype(np.float32), target.astype(np.float32), original_length


class SupervisedSnippetDataset(Dataset):
    """Load a disjoint diagnostic train split with its own frame annotations."""

    def __init__(self, csv_path: str, gt_path: str, visual_length: int, expected_width: int) -> None:
        self.frame = read_csv(csv_path).reset_index(drop=True)
        self.gt = np.asarray(np.load(gt_path), dtype=np.float32).reshape(-1)
        self.visual_length = int(visual_length)
        self.expected_width = int(expected_width)
        self.frame_offsets: list[tuple[int, int]] = []
        offset = 0
        for row in self.frame.itertuples(index=False):
            feature = load_clip_feature(str(row.path))
            if feature.shape[1] != self.expected_width:
                raise ValueError(f"{row.path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
            frame_count = int(feature.shape[0]) * 16
            self.frame_offsets.append((offset, offset + frame_count))
            offset += frame_count
        if offset != len(self.gt):
            raise ValueError(
                "diagnostic train GT does not align to its CSV/features: "
                f"features imply {offset} frames, GT has {len(self.gt)}"
            )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_clip_feature(path)
        left, right = self.frame_offsets[index]
        frame_target = self.gt[left:right]
        if frame_target.size != feature.shape[0] * 16:
            raise RuntimeError(f"{path}: stored GT range is inconsistent with its feature length")
        snippet_target = (frame_target.reshape(feature.shape[0], 16).max(axis=1) > 0).astype(np.float32)
        feature, snippet_target, length = process_feature_and_target(feature, snippet_target, self.visual_length)
        return torch.from_numpy(feature), torch.from_numpy(snippet_target), int(length)
