"""VadCLIP-consistent XD inference with a frozen label-free reliability gate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from common import XD_LABELS, base_key, load_hidden, resample_scores, test_chunk_lengths
from models import split_concat
from reliability import ReliabilityConfig, manifest_map, reliability_map
from xd_evaluation import softmax_numpy, summarize_records


@dataclass(frozen=True)
class ReliabilityRuntime:
    """Frozen artifacts needed to compute q for one concat test video."""

    hidden_paths: dict[str, str]
    normal_mean: np.ndarray
    normal_std: np.ndarray
    config: ReliabilityConfig


def runtime_from_contract(contract: dict[str, Any], hidden_manifest: str) -> ReliabilityRuntime:
    """Load the exact training-only calibration persisted by the selector."""
    required = {
        "normal_mean_path", "normal_std_path", "seed_top_p", "expand_top_p",
        "normal_score_quantile", "score_temperature", "sigma_min", "normal_score_threshold",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"neuron JSON lacks reliability fields: {sorted(missing)}")
    normal_mean = np.asarray(np.load(contract["normal_mean_path"], allow_pickle=False), dtype=np.float32)
    normal_std = np.asarray(np.load(contract["normal_std_path"], allow_pickle=False), dtype=np.float32)
    if normal_mean.ndim != 2 or normal_std.shape != normal_mean.shape:
        raise ValueError("reliability normal statistics must be matching [layers,dimensions] arrays")
    config = ReliabilityConfig(
        seed_top_p=float(contract["seed_top_p"]), expand_top_p=float(contract["expand_top_p"]),
        normal_score_quantile=float(contract["normal_score_quantile"]),
        score_temperature=float(contract["score_temperature"]), sigma_min=float(contract["sigma_min"]),
        normal_score_threshold=float(contract["normal_score_threshold"]),
    )
    return ReliabilityRuntime(manifest_map(hidden_manifest), normal_mean, normal_std, config)


def reliability_for_video(runtime: ReliabilityRuntime, source_path: str, baseline_prob1: np.ndarray, output_length: int) -> np.ndarray:
    """Compute q from frozen baseline scores and a matching shared hidden cache."""
    key = base_key(source_path)
    hidden_path = runtime.hidden_paths.get(key)
    if hidden_path is None:
        raise FileNotFoundError(f"test hidden manifest has no artifact for {key} ({source_path})")
    hidden, _metadata = load_hidden(hidden_path)
    q_hidden, _aligned, _seeds, _similarity = reliability_map(
        hidden, baseline_prob1, runtime.normal_mean, runtime.normal_std, runtime.config
    )
    return resample_scores(q_hidden, output_length)


def _masked_inputs(item, visual_length: int, device: torch.device):
    visual = item[0].squeeze(0)
    original_length = int(item[1].item())
    source_path, label = str(item[2][0]), str(item[3][0])
    if original_length < visual_length:
        visual = visual.unsqueeze(0)
    visual = visual.to(device, non_blocking=True)
    lengths = torch.as_tensor(test_chunk_lengths(original_length, visual_length), dtype=torch.int64, device=device)
    padding_mask = torch.zeros((len(lengths), visual_length), dtype=torch.bool, device=device)
    for index, length in enumerate(lengths.tolist()):
        if length < visual_length:
            padding_mask[index, max(length, 0):] = True
    return visual, original_length, source_path, label, lengths, padding_mask


def infer_reliability_gated_item(
    model,
    item,
    visual_length: int,
    device: torch.device,
    runtime: ReliabilityRuntime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, np.ndarray]:
    """Gate only the residual logit delta, preserving base scores when q is zero."""
    visual, original_length, source_path, label, lengths, padding_mask = _masked_inputs(item, visual_length, device)
    prompts = list(XD_LABELS.values())
    with torch.no_grad():
        neuron, clip = split_concat(visual, model.neuron_width, model.clip_dim)
        correction = model.neuron_to_clip(model.neuron_norm(neuron))
        enhanced_clip = clip + model.residual_gate().to(clip.dtype) * correction.to(clip.dtype)
        _base_text, base_logits1, base_logits2 = model.base(clip, padding_mask, prompts, lengths)
        _residual_text, residual_logits1, residual_logits2 = model.base(enhanced_clip, padding_mask, prompts, lengths)
    base_logits1 = base_logits1.reshape(-1, base_logits1.shape[-1])[:original_length]
    base_logits2 = base_logits2.reshape(-1, base_logits2.shape[-1])[:original_length]
    residual_logits1 = residual_logits1.reshape(-1, residual_logits1.shape[-1])[:original_length]
    residual_logits2 = residual_logits2.reshape(-1, residual_logits2.shape[-1])[:original_length]
    base_prob1 = torch.sigmoid(base_logits1.squeeze(-1)).detach().cpu().numpy().astype(np.float32)
    q = reliability_for_video(runtime, source_path, base_prob1, original_length)
    q_tensor = torch.from_numpy(q).to(device=device, dtype=base_logits1.dtype).unsqueeze(-1)
    logits1 = base_logits1 + q_tensor * (residual_logits1 - base_logits1)
    logits2 = base_logits2 + q_tensor * (residual_logits2 - base_logits2)
    prob1 = torch.sigmoid(logits1.squeeze(-1)).detach().cpu().numpy().astype(np.float32)
    prob2 = (1.0 - logits2.softmax(dim=-1)[:, 0]).detach().cpu().numpy().astype(np.float32)
    return prob1, prob2, logits2.detach().cpu().numpy().astype(np.float32), source_path, label, q.astype(np.float32)


def run_reliability_evaluation(
    model,
    loader,
    visual_length: int,
    device: torch.device,
    runtime: ReliabilityRuntime,
    gt_path: str,
    segment_path: str,
    label_path: str,
    vadclip_root: str,
    description: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str]], dict[str, object]]:
    """Match the official XD metrics after applying the pre-fixed q gate."""
    model.to(device).eval()
    records = []
    for item in tqdm(loader, desc=description, unit="video"):
        result = infer_reliability_gated_item(model, item, visual_length, device, runtime)
        records.append(result[:5])
    return records, summarize_records(records, gt_path, segment_path, label_path, vadclip_root)


def save_reliability_record(path: Path, record: tuple[np.ndarray, np.ndarray, np.ndarray, str, str, np.ndarray]) -> None:
    """Write one resumable test artifact, including q for post-run auditing."""
    np.savez_compressed(
        path, prob1=record[0], prob2=record[1], logits2=record[2],
        source_path=np.asarray(record[3]), label=np.asarray(record[4]), reliability=record[5],
    )


def load_reliability_record(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, np.ndarray]:
    """Validate a completed artifact before a resumable test reuses it."""
    with np.load(path, allow_pickle=False) as artifact:
        required = {"prob1", "prob2", "logits2", "source_path", "label", "reliability"}
        if not required.issubset(artifact.files):
            raise ValueError(f"{path}: incomplete reliability test artifact")
        return (
            np.asarray(artifact["prob1"], dtype=np.float32), np.asarray(artifact["prob2"], dtype=np.float32),
            np.asarray(artifact["logits2"], dtype=np.float32), str(artifact["source_path"].item()),
            str(artifact["label"].item()), np.asarray(artifact["reliability"], dtype=np.float32),
        )


def reliability_summary(records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str, np.ndarray]]) -> dict[str, float]:
    """Small, label-free audit of how often the residual delta was permitted."""
    q = np.concatenate([record[5] for record in records])
    return {"mean": float(q.mean()), "p95": float(np.quantile(q, 0.95)), "max": float(q.max())}
