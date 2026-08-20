"""Baseline-agnostic pseudo-score ranking supervision for residual injection.

The module consumes only a training CSV and frozen-baseline pseudo scores.  It
does not inspect test labels, test predictions, or baseline internals, so the
same objective can be used by both VadCLIP and DSANet wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from tqdm import tqdm

from common import (
    feature_key,
    grouped_rows,
    is_normal_label,
    load_clip_feature,
    process_train_scores,
    read_csv,
)


def _pseudo_rows(path: str) -> dict[str, tuple[str, str]]:
    frame = pd.read_csv(path)
    missing = {"key", "label", "score_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing pseudo-score columns: {sorted(missing)}")
    rows: dict[str, tuple[str, str]] = {}
    for _, row in frame.iterrows():
        key = str(row["key"])
        if key in rows:
            raise ValueError(f"{path} has duplicate pseudo-score key {key!r}")
        rows[key] = (str(row["label"]), str(row["score_path"]))
    return rows


class PseudoRankingTargets:
    """Map each abnormal concat-feature path to its aligned teacher scores.

    ``score_vadclip_pseudo.py`` stores one score array per original video
    group, whereas the concat train CSV may contain ``__0``, ``__1``, ...
    variants.  The constructor restores those exact slices in chunk order and
    validates that their lengths exhaust the score array.  This prevents a
    silent temporal misalignment during ranking supervision.
    """

    def __init__(self, dataset: str, train_csv: str, pseudo_csv: str, visual_length: int) -> None:
        self.visual_length = int(visual_length)
        if self.visual_length <= 0:
            raise ValueError("visual_length must be positive")
        pseudo_by_key = _pseudo_rows(pseudo_csv)
        frame = read_csv(train_csv)
        self._targets: dict[str, np.ndarray] = {}
        abnormal_groups = 0
        for key, group in tqdm(grouped_rows(frame).items(), desc="align ranking pseudo scores", unit="video"):
            labels = set(group["label"].astype(str))
            if len(labels) != 1:
                raise ValueError(f"{train_csv}: video {key!r} has inconsistent labels {sorted(labels)}")
            label = next(iter(labels))
            if is_normal_label(dataset, label):
                continue
            if key not in pseudo_by_key:
                raise KeyError(f"{pseudo_csv}: missing abnormal video {key!r} required by {train_csv}")
            pseudo_label, score_path = pseudo_by_key[key]
            if pseudo_label != label:
                raise ValueError(
                    f"{key}: pseudo label {pseudo_label!r} differs from concat-train label {label!r}"
                )
            scores = np.asarray(np.load(score_path), dtype=np.float32).reshape(-1)
            if scores.size == 0 or not np.isfinite(scores).all():
                raise ValueError(f"{score_path}: expected a finite non-empty pseudo-score vector")
            offset = 0
            for _, row in group.iterrows():
                path = str(row["path"])
                length = int(load_clip_feature(path).shape[0])
                stop = offset + length
                if stop > scores.size:
                    raise ValueError(
                        f"{key}: concat-feature lengths exceed pseudo-score length "
                        f"({stop} > {scores.size})"
                    )
                target, _valid_length = process_train_scores(scores[offset:stop], self.visual_length)
                target_key = feature_key(path)
                if target_key in self._targets:
                    raise ValueError(f"duplicate concat feature name while aligning pseudo scores: {target_key}")
                self._targets[target_key] = target
                offset = stop
            if offset != scores.size:
                raise ValueError(
                    f"{key}: concat-feature lengths total {offset}, but pseudo-score length is {scores.size}; "
                    "the pseudo CSV must be produced from the same train CSV before concat construction"
                )
            abnormal_groups += 1
        if not self._targets:
            raise RuntimeError("no abnormal concat features received pseudo-score ranking targets")
        print(
            f"ranking pseudo targets: abnormal_videos={abnormal_groups}, concat_features={len(self._targets)}",
            flush=True,
        )

    def target_for(self, path: str) -> np.ndarray:
        key = feature_key(path)
        target = self._targets.get(key)
        if target is None:
            raise KeyError(f"missing pseudo-score target for abnormal concat feature {path}")
        return target.copy()


@dataclass(frozen=True)
class RankingStats:
    abnormal_videos: int
    normal_videos: int
    mean_confidence: float


def _rank_count(valid_length: int, top_p: float) -> int:
    requested = max(1, int(np.ceil(top_p * valid_length)))
    return min(requested, valid_length // 2)


def temporal_ranking_terms(
    predicted_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    lengths: torch.Tensor,
    abnormal_mask: torch.Tensor,
    top_p: float,
    intra_margin: float,
    cross_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, RankingStats]:
    """Return confidence-weighted intra-video and hard-normal ranking losses.

    For every abnormal video, teacher top/bottom ``top_p`` snippets define P
    and N.  The student is asked to rank P above N and above the student's own
    high-score normal snippets H.  Teacher confidence is the P--N gap divided
    by that video's score range, making the weighting invariant to each
    baseline's sigmoid/logit calibration.
    """
    if predicted_scores.ndim != 2 or teacher_scores.ndim != 2:
        raise ValueError("predicted_scores and teacher_scores must have shape [batch, time]")
    if predicted_scores.shape != teacher_scores.shape:
        raise ValueError("predicted_scores and teacher_scores must have identical shape")
    if lengths.ndim != 1 or lengths.shape[0] != predicted_scores.shape[0]:
        raise ValueError("lengths must have one entry per batch element")
    if abnormal_mask.shape != lengths.shape:
        raise ValueError("abnormal_mask must have one entry per batch element")

    zero = predicted_scores.new_zeros(())
    positive_means: list[torch.Tensor] = []
    negative_means: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    normal_hard_means: list[torch.Tensor] = []
    for index in range(predicted_scores.shape[0]):
        valid = max(1, min(int(lengths[index]), predicted_scores.shape[1]))
        student = predicted_scores[index, :valid]
        if bool(abnormal_mask[index]):
            teacher = teacher_scores[index, :valid]
            count = _rank_count(valid, top_p)
            if count <= 0:
                continue
            top_indices = torch.topk(teacher, count, largest=True, sorted=False).indices
            bottom_indices = torch.topk(teacher, count, largest=False, sorted=False).indices
            teacher_gap = teacher[top_indices].mean() - teacher[bottom_indices].mean()
            teacher_range = teacher.max() - teacher.min()
            confidence = (teacher_gap / teacher_range.clamp_min(1e-6)).clamp(0.0, 1.0)
            positive_means.append(student[top_indices].mean())
            negative_means.append(student[bottom_indices].mean())
            confidences.append(confidence)
        else:
            hard_count = max(1, int(np.ceil(top_p * valid)))
            normal_hard_means.append(torch.topk(student, hard_count, largest=True).values.mean())

    if not positive_means:
        return zero, zero, RankingStats(0, len(normal_hard_means), 0.0)
    positive = torch.stack(positive_means)
    negative = torch.stack(negative_means)
    confidence = torch.stack(confidences)
    confidence_sum = confidence.sum()
    if float(confidence_sum.detach()) <= 0.0:
        intra = zero
    else:
        intra = (confidence * functional.softplus(intra_margin - positive + negative)).sum() / confidence_sum
    if normal_hard_means and float(confidence_sum.detach()) > 0.0:
        hard_normal = torch.stack(normal_hard_means).mean()
        cross = (confidence * functional.softplus(cross_margin - positive + hard_normal)).sum() / confidence_sum
    else:
        cross = zero
    return intra, cross, RankingStats(
        abnormal_videos=len(positive_means),
        normal_videos=len(normal_hard_means),
        mean_confidence=float(confidence.detach().mean().cpu()),
    )


def dual_branch_temporal_ranking_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    teacher_scores: torch.Tensor,
    lengths: torch.Tensor,
    abnormal_mask: torch.Tensor,
    top_p: float,
    intra_margin: float,
    cross_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RankingStats]:
    """Rank both official anomaly outputs without altering either base loss."""
    score1 = torch.sigmoid(logits1.reshape(logits1.shape[0], logits1.shape[1]))
    score2 = 1.0 - logits2.softmax(dim=-1)[..., 0]
    intra1, cross1, stats = temporal_ranking_terms(
        score1, teacher_scores, lengths, abnormal_mask, top_p, intra_margin, cross_margin
    )
    intra2, cross2, _ = temporal_ranking_terms(
        score2, teacher_scores, lengths, abnormal_mask, top_p, intra_margin, cross_margin
    )
    return (intra1 + intra2) * 0.5, (cross1 + cross2) * 0.5, (intra1 + intra2 + cross1 + cross2) * 0.5, stats
