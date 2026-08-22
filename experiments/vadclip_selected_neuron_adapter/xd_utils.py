"""VadCLIP-consistent XD losses, inference, and metric calculation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.metrics import average_precision_score, roc_auc_score

from common import add_vadclip_source
from online_data import OnlineVideoDataset, VideoSample, process_test_feature, test_chunk_lengths


XD_LABELS = {
    "A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot",
    "B5": "abuse", "B6": "car accident", "G": "explosion",
}


def prompt_text() -> list[str]:
    return list(XD_LABELS.values())


def label_tensor(label: str, device: torch.device) -> torch.Tensor:
    """Match one row of the official XD ``get_batch_label`` output."""
    prompts = prompt_text()
    lookup = {name: index for index, name in enumerate(prompts)}
    output = torch.zeros((1, len(prompts)), dtype=torch.float32, device=device)
    for code in str(label).split("-"):
        if code not in XD_LABELS:
            raise ValueError(f"unrecognised XD label {label!r}")
        output[0, lookup[XD_LABELS[code]]] = 1.0
    return output


def clas2_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The official XD binary MIL loss, with the same top-k definition."""
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    instance_logits = []
    for index in range(probabilities.shape[0]):
        valid = max(1, min(int(lengths[index]), probabilities.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(probabilities[index, :valid].topk(count).values.mean())
    return functional.binary_cross_entropy(torch.stack(instance_logits), 1.0 - labels[:, 0])


def clasm_loss(logits: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """The official XD language MIL loss, unchanged."""
    normalized_labels = labels / labels.sum(dim=1, keepdim=True)
    instance_logits = []
    for index in range(logits.shape[0]):
        valid = max(1, min(int(lengths[index]), logits.shape[1]))
        count = max(1, min(int(valid / 16 + 1), valid))
        instance_logits.append(logits[index, :valid].topk(count, dim=0).values.mean(dim=0))
    return -torch.mean(torch.sum(normalized_labels * functional.log_softmax(torch.stack(instance_logits), dim=1), dim=1))


def text_separation_loss(text_features: torch.Tensor) -> torch.Tensor:
    normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
    loss = torch.zeros((), device=text_features.device)
    for index in range(1, text_features.shape[0]):
        anomaly = text_features[index] / text_features[index].norm(dim=-1, keepdim=True)
        loss = loss + torch.abs(normal @ anomaly)
    return loss / 6.0


def padding_mask(lengths: torch.Tensor, visual_length: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((len(lengths), visual_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths.tolist()):
        if length < visual_length:
            mask[index, max(int(length), 0):] = True
    return mask


def infer_sample(model, sample: VideoSample, visual_length: int, frame_batch_size: int, device: torch.device):
    """Online visual forward followed by the exact official XD scoring path."""
    frames = sample.frames.to(device, non_blocking=True)
    features = model.encode_feature_anchored_sequence(
        frames, sample.source_feature.to(device, non_blocking=True), frame_batch_size
    )
    visual, original_length = process_test_feature(features, visual_length)
    lengths = test_chunk_lengths(original_length, visual_length).to(device)
    mask = padding_mask(lengths, visual_length, device)
    _text, logits1, logits2 = model(visual, mask, prompt_text(), lengths)
    logits1 = logits1.reshape(-1, logits1.shape[-1])[:original_length]
    logits2 = logits2.reshape(-1, logits2.shape[-1])[:original_length]
    prob1 = torch.sigmoid(logits1.squeeze(-1)).detach().cpu().numpy().astype(np.float32)
    prob2 = (1.0 - logits2.softmax(dim=-1)[:, 0]).detach().cpu().numpy().astype(np.float32)
    return prob1, prob2, logits2.detach().cpu().numpy().astype(np.float32), sample.source_path, sample.label


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def summarize_records(records, gt_path: str, segment_path: str, label_path: str, vadclip_root: str) -> dict[str, object]:
    if not records:
        raise RuntimeError("no XD records were evaluated")
    gt = np.load(gt_path)
    prob1 = np.concatenate([record[0] for record in records])
    prob2 = np.concatenate([record[1] for record in records])
    if len(prob1) * 16 != len(gt):
        raise ValueError(f"XD frame alignment failed: prediction frames={len(prob1) * 16}, GT frames={len(gt)}")
    metrics: dict[str, object] = {
        "videos": len(records),
        "snippet_scores": int(len(prob1)),
        "frame_scores": int(len(prob1) * 16),
        "roc_auc_logits1": float(roc_auc_score(gt, np.repeat(prob1, 16))),
        "ap_logits1": float(average_precision_score(gt, np.repeat(prob1, 16))),
        "roc_auc_logits2": float(roc_auc_score(gt, np.repeat(prob2, 16))),
        "ap_logits2": float(average_precision_score(gt, np.repeat(prob2, 16))),
    }
    add_vadclip_source(vadclip_root)
    from utils.xd_detectionMAP import getDetectionMAP

    segments = np.load(segment_path, allow_pickle=True)
    labels = np.load(label_path, allow_pickle=True)
    element_logits = [np.repeat(softmax_numpy(record[2]), 16, axis=0) for record in records]
    detection_map, ious = getDetectionMAP(element_logits, segments, labels, excludeNormal=False)
    metrics["detection_map"] = {f"iou_{iou:.1f}": float(value) for iou, value in zip(ious, detection_map)}
    metrics["detection_map_average"] = float(np.mean(detection_map))
    return metrics


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


def save_metrics(path: str | Path, metrics: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def append_history(path: Path, row: dict[str, object]) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_online_dataset(source_csv: str, hidden_manifest: str, vadclip_root: str) -> OnlineVideoDataset:
    return OnlineVideoDataset(source_csv, hidden_manifest, vadclip_root)
