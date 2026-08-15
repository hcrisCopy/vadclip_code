"""VadCLIP-consistent UCF inference and metric helpers for the residual model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import UCF_TEST_LABELS, load_clip_feature, process_test_feature, read_csv, test_chunk_lengths


class UCFConcatTestDataset(Dataset):
    def __init__(self, csv_path: str, visual_length: int, expected_width: int) -> None:
        self.frame = read_csv(csv_path).reset_index(drop=True)
        self.visual_length = int(visual_length)
        self.expected_width = int(expected_width)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.loc[index]
        path = str(row["path"])
        feature = load_clip_feature(path)
        if feature.shape[1] != self.expected_width:
            raise ValueError(f"{path}: expected {self.expected_width}D concat feature, got {feature.shape[1]}D")
        split, length = process_test_feature(feature, self.visual_length)
        return torch.from_numpy(split), int(length), path, str(row["label"])


def add_vadclip_source(vadclip_root: str) -> None:
    source = str(Path(vadclip_root) / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def build_test_loader(csv_path: str, visual_length: int, expected_width: int, num_workers: int) -> DataLoader:
    dataset = UCFConcatTestDataset(csv_path, visual_length, expected_width)
    # No pin_memory argument in the official UCF test loader; preserve that
    # default while retaining a configurable worker count for resumed tests.
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)


def infer_item(model, item, visual_length: int, prompt_text: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
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
    prob1 = torch.sigmoid(logits1.squeeze(-1)).detach().cpu().numpy().astype(np.float32)
    prob2 = (1.0 - logits2.softmax(dim=-1)[:, 0]).detach().cpu().numpy().astype(np.float32)
    return prob1, prob2, logits2.detach().cpu().numpy().astype(np.float32), path, label


def summarize_records(records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]], gt_path: str, segment_path: str, label_path: str, vadclip_root: str) -> dict[str, object]:
    if not records:
        raise RuntimeError("no UCF test records were evaluated")
    gt = np.load(gt_path)
    prob1 = np.concatenate([record[0] for record in records])
    prob2 = np.concatenate([record[1] for record in records])
    expected = len(prob1) * 16
    if expected != len(gt):
        raise ValueError(f"UCF frame alignment failed: prediction frames={expected}, gt frames={len(gt)}")
    metrics: dict[str, object] = {
        "videos": len(records), "snippet_scores": int(len(prob1)), "frame_scores": int(expected),
        "roc_auc_logits1": float(roc_auc_score(gt, np.repeat(prob1, 16))),
        "ap_logits1": float(average_precision_score(gt, np.repeat(prob1, 16))),
        "roc_auc_logits2": float(roc_auc_score(gt, np.repeat(prob2, 16))),
        "ap_logits2": float(average_precision_score(gt, np.repeat(prob2, 16))),
    }
    add_vadclip_source(vadclip_root)
    from utils.ucf_detectionMAP import getDetectionMAP

    segments = np.load(segment_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    element_logits = [np.repeat(record[2], 16, axis=0) for record in records]
    detection_map, ious = getDetectionMAP(element_logits, segments, labels, excludeNormal=False)
    metrics["detection_map"] = {f"iou_{iou:.1f}": float(value) for iou, value in zip(ious, detection_map)}
    metrics["detection_map_average"] = float(np.mean(detection_map))
    return metrics


def run_evaluation(model, loader: DataLoader, visual_length: int, device: torch.device, gt_path: str, segment_path: str, label_path: str, vadclip_root: str, description: str = "VadCLIP residual evaluation") -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]], dict[str, object]]:
    model.to(device).eval()
    prompts = list(UCF_TEST_LABELS.values())
    records = []
    for item in tqdm(loader, desc=description, unit="video"):
        records.append(infer_item(model, item, visual_length, prompts, device))
    metrics = summarize_records(records, gt_path, segment_path, label_path, vadclip_root)
    return records, metrics


def print_metrics(metrics: dict[str, object]) -> None:
    print(
        "UCF metrics | "
        f"AUC1={metrics['roc_auc_logits1']:.6f} AP1={metrics['ap_logits1']:.6f} | "
        f"AUC2={metrics['roc_auc_logits2']:.6f} AP2={metrics['ap_logits2']:.6f}",
        flush=True,
    )
    for key, value in metrics["detection_map"].items():
        print(f"mAP@{key.removeprefix('iou_')}={value:.2f}%", flush=True)
    print(f"average detection mAP={metrics['detection_map_average']:.2f}%", flush=True)


def save_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
