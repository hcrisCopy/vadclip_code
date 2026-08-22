"""XD-Violence data, checkpoint and evaluation helpers for M1/M2.

These helpers reproduce the official VadCLIP XD loader and metrics while
checking the new 768+512 concat-feature contract.  They deliberately import
only the untouched ``VadCLIP/src`` evaluator at final metric time.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


XD_LABELS = {
    "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
    "B5": "abuse", "B6": "car accident", "G": "explosion",
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def clean_dir(path: str | Path) -> Path:
    """Delete only an explicitly named output directory for ``--clean``."""
    target = Path(path)
    if target in {Path("."), Path(".."), Path("/")} or target.name in {"", "."}:
        raise ValueError("--clean refuses to remove the current or root directory")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, content: dict) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"path", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def load_feature(path: str | Path) -> np.ndarray:
    artifact = np.load(path, allow_pickle=True)
    if isinstance(artifact, np.lib.npyio.NpzFile):
        key = "features" if "features" in artifact.files else artifact.files[0]
        feature = artifact[key]
    else:
        feature = artifact
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] <= 0:
        raise ValueError(f"{path}: expected non-empty [T,D] feature, got {feature.shape}")
    return feature


def process_train_feature(feature: np.ndarray, visual_length: int) -> tuple[np.ndarray, int]:
    """Match VadCLIP ``utils.tools.process_feat`` exactly."""
    feature = np.asarray(feature, dtype=np.float32)
    original_length = int(feature.shape[0])
    if original_length > visual_length:
        reduced = np.zeros((visual_length, feature.shape[1]), dtype=np.float32)
        boundaries = np.linspace(0, original_length, visual_length + 1, dtype=np.int32)
        for index in range(visual_length):
            left, right = boundaries[index], boundaries[index + 1]
            reduced[index] = feature[left:right].mean(axis=0) if left != right else feature[left]
        return reduced, visual_length
    if original_length < visual_length:
        feature = np.pad(feature, ((0, visual_length - original_length), (0, 0)), mode="constant")
    return feature.astype(np.float32), original_length


def process_test_feature(feature: np.ndarray, visual_length: int) -> tuple[np.ndarray, int]:
    """Match VadCLIP ``utils.tools.process_split``, including final padding."""
    feature = np.asarray(feature, dtype=np.float32)
    original_length = int(feature.shape[0])
    if original_length < visual_length:
        return np.pad(feature, ((0, visual_length - original_length), (0, 0)), mode="constant"), original_length
    chunks = []
    for index in range(int(original_length / visual_length) + 1):
        part = feature[index * visual_length:index * visual_length + visual_length]
        if part.shape[0] < visual_length:
            part = np.pad(part, ((0, visual_length - part.shape[0]), (0, 0)), mode="constant")
        chunks.append(part.reshape(1, visual_length, feature.shape[1]))
    return np.concatenate(chunks, axis=0), original_length


def test_chunk_lengths(original_length: int, visual_length: int) -> np.ndarray:
    """Match the per-chunk valid-length construction in ``xd_test.py``."""
    remaining, lengths = int(original_length), []
    for index in range(int(original_length / visual_length) + 1):
        if index == 0 and original_length < visual_length:
            lengths.append(original_length)
        elif index == 0 and original_length > visual_length:
            lengths.append(visual_length)
            remaining -= visual_length
        elif remaining > visual_length:
            lengths.append(visual_length)
            remaining -= visual_length
        else:
            lengths.append(remaining)
    return np.asarray(lengths, dtype=np.int64)


class XDConcatTrainDataset(Dataset):
    """Official XD train CSV semantics with 1280D feature validation."""

    def __init__(self, csv_path: str, visual_length: int, expected_width: int) -> None:
        self.frame = read_csv(csv_path).reset_index(drop=True)
        self.visual_length, self.expected_width = int(visual_length), int(expected_width)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_feature(path)
        if feature.shape[1] != self.expected_width:
            raise ValueError(f"{path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
        feature, length = process_train_feature(feature, self.visual_length)
        return torch.from_numpy(feature), str(row["label"]), int(length)


class XDConcatTestDataset(Dataset):
    """Official XD test CSV order with resumable 1280D input validation."""

    def __init__(self, csv_path: str, visual_length: int, expected_width: int) -> None:
        self.frame = read_csv(csv_path).reset_index(drop=True)
        self.visual_length, self.expected_width = int(visual_length), int(expected_width)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_feature(path)
        if feature.shape[1] != self.expected_width:
            raise ValueError(f"{path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
        split, length = process_test_feature(feature, self.visual_length)
        return torch.from_numpy(split), int(length), path, str(row["label"])


def build_train_loader(csv_path: str, visual_length: int, expected_width: int, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        XDConcatTrainDataset(csv_path, visual_length, expected_width),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def build_test_loader(csv_path: str, visual_length: int, expected_width: int, num_workers: int) -> DataLoader:
    return DataLoader(
        XDConcatTestDataset(csv_path, visual_length, expected_width),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )


def label_tensor(labels: list[str], device: torch.device) -> torch.Tensor:
    prompts = list(XD_LABELS.values())
    lookup = {name: index for index, name in enumerate(prompts)}
    target = torch.zeros((len(labels), len(prompts)), dtype=torch.float32, device=device)
    for row, text in enumerate(labels):
        matched = False
        for code in str(text).split("-"):
            # ``0`` tokens in XD lists are padding, not action classes.
            if code in XD_LABELS:
                target[row, lookup[XD_LABELS[code]]] = 1.0
                matched = True
        if not matched:
            raise ValueError(f"unrecognised XD label {text!r}")
    return target


def clas2_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    instances = []
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for index in range(probabilities.shape[0]):
        valid = max(1, min(int(lengths[index]), probabilities.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instances.append(probabilities[index, :valid].topk(count).values.mean())
    return torch.nn.functional.binary_cross_entropy(torch.stack(instances), 1.0 - labels[:, 0])


def clasm_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    instances = []
    labels = labels / labels.sum(dim=1, keepdim=True)
    for index in range(logits.shape[0]):
        valid = max(1, min(int(lengths[index]), logits.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instances.append(logits[index, :valid].topk(count, dim=0).values.mean(dim=0))
    return -torch.mean(torch.sum(labels * torch.nn.functional.log_softmax(torch.stack(instances), dim=1), dim=1))


def text_separation_loss(text_features: torch.Tensor) -> torch.Tensor:
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    loss = torch.zeros((), device=text_features.device)
    for index in range(1, text_features.shape[0]):
        anomaly = text_features[index] / text_features[index].norm(dim=-1, keepdim=True)
        loss = loss + torch.abs(normal @ anomaly)
    return loss / 6.0


def infer_item(model, item, visual_length: int, prompt_text: list[str], device: torch.device):
    """Official XD inference path, retained for validation and final test."""
    visual = item[0].squeeze(0)
    original_length = int(item[1].item())
    path, label = str(item[2][0]), str(item[3][0])
    if original_length < visual_length:
        visual = visual.unsqueeze(0)
    visual = visual.to(device, non_blocking=True)
    lengths = torch.as_tensor(test_chunk_lengths(original_length, visual_length), dtype=torch.int64, device=device)
    padding_mask = torch.zeros((len(lengths), visual_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths.tolist()):
        if length < visual_length:
            padding_mask[index, max(length, 0):] = True
    with torch.no_grad():
        _text, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
    logits1 = logits1.reshape(-1, logits1.shape[-1])[:original_length]
    logits2 = logits2.reshape(-1, logits2.shape[-1])[:original_length]
    prob1 = torch.sigmoid(logits1.squeeze(-1)).cpu().numpy().astype(np.float32)
    prob2 = (1.0 - logits2.softmax(dim=-1)[:, 0]).cpu().numpy().astype(np.float32)
    return prob1, prob2, logits2.cpu().numpy().astype(np.float32), path, label


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=1, keepdims=True)


def _add_vadclip_source(vadclip_root: str) -> None:
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def summarize_records(records, gt_path: str, segment_path: str, label_path: str, vadclip_root: str) -> dict[str, object]:
    if not records:
        raise RuntimeError("no XD test records were evaluated")
    ground_truth = np.load(gt_path)
    prob1 = np.concatenate([record[0] for record in records])
    prob2 = np.concatenate([record[1] for record in records])
    if len(prob1) * 16 != len(ground_truth):
        raise ValueError(f"XD frame alignment failed: prediction frames={len(prob1) * 16}, gt frames={len(ground_truth)}")
    _add_vadclip_source(vadclip_root)
    from utils.xd_detectionMAP import getDetectionMAP

    detection_map, ious = getDetectionMAP(
        [np.repeat(softmax_numpy(record[2]), 16, axis=0) for record in records],
        np.load(segment_path, allow_pickle=True),
        np.load(label_path, allow_pickle=True),
        excludeNormal=False,
    )
    return {
        "videos": len(records),
        "snippet_scores": int(len(prob1)),
        "frame_scores": int(len(prob1) * 16),
        "roc_auc_logits1": float(roc_auc_score(ground_truth, np.repeat(prob1, 16))),
        "ap_logits1": float(average_precision_score(ground_truth, np.repeat(prob1, 16))),
        "roc_auc_logits2": float(roc_auc_score(ground_truth, np.repeat(prob2, 16))),
        "ap_logits2": float(average_precision_score(ground_truth, np.repeat(prob2, 16))),
        "detection_map": {f"iou_{iou:.1f}": float(value) for iou, value in zip(ious, detection_map)},
        "detection_map_average": float(np.mean(detection_map)),
    }


def run_evaluation(model, loader: DataLoader, visual_length: int, device: torch.device, gt_path: str, segment_path: str, label_path: str, vadclip_root: str, description: str):
    model.to(device).eval()
    records = []
    for item in tqdm(loader, desc=description, unit="video"):
        records.append(infer_item(model, item, visual_length, list(XD_LABELS.values()), device))
    return records, summarize_records(records, gt_path, segment_path, label_path, vadclip_root)


def print_metrics(metrics: dict[str, object]) -> None:
    print(
        "XD metrics | "
        f"AUC1={metrics['roc_auc_logits1']:.6f} AP1={metrics['ap_logits1']:.6f} | "
        f"AUC2={metrics['roc_auc_logits2']:.6f} AP2={metrics['ap_logits2']:.6f}",
        flush=True,
    )
    for key, value in metrics["detection_map"].items():
        print(f"mAP@{key.removeprefix('iou_')}={value:.2f}%", flush=True)
    print(f"average detection mAP={metrics['detection_map_average']:.2f}%", flush=True)
