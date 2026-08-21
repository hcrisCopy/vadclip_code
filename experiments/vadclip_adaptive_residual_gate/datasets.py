"""Original global-768 train datasets without pseudo supervision or frame labels."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from shared import add_injection_source

add_injection_source()
from common import is_normal_label, load_clip_feature, process_train_feature, read_csv


class AdaptiveConcatTrainDataset(Dataset):
    """Load 1280D concat features using the ordinary VadCLIP train reduction."""

    def __init__(
        self, csv_path: str, visual_length: int, expected_width: int, dataset: str, normal: bool | None = None
    ) -> None:
        frame = read_csv(csv_path)
        if normal is not None:
            mask = frame["label"].astype(str).map(lambda label: is_normal_label(dataset, label))
            frame = frame.loc[mask if normal else ~mask]
        self.frame = frame.reset_index(drop=True)
        self.visual_length, self.expected_width = int(visual_length), int(expected_width)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_clip_feature(path)
        if feature.shape[1] != self.expected_width:
            raise ValueError(f"{path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
        feature, length = process_train_feature(feature, self.visual_length)
        return torch.from_numpy(feature), str(row["label"]), int(length)
