"""Raw-video loader aligned to the already extracted CLIP-hidden manifest."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from common import add_vadclip_source, base_key, read_csv, resolve_existing_path


@dataclass(frozen=True)
class VideoSample:
    """One source feature row and its exact raw frames used by hidden extraction."""

    frames: torch.Tensor
    label: str
    source_path: str
    video_path: str


@dataclass(frozen=True)
class ManifestEntry:
    video_path: Path
    hidden_path: Path


def build_clip_preprocess(vadclip_root: str):
    """Use the unmodified VadCLIP CLIP preprocessing definition (224px ViT-B/16)."""
    add_vadclip_source(vadclip_root)
    from clip.clip import _transform

    return _transform(224)


def load_manifest(manifest_path: str) -> dict[str, ManifestEntry]:
    manifest_file = Path(manifest_path)
    frame = pd.read_csv(manifest_file)
    missing = {"key", "hidden_path", "video_path"} - set(frame.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {sorted(missing)}")
    entries: dict[str, ManifestEntry] = {}
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="index hidden manifest", unit="video"):
        key = str(row["key"])
        if key in entries:
            raise ValueError(f"{manifest_path}: duplicate hidden key {key!r}")
        entries[key] = ManifestEntry(
            video_path=resolve_existing_path(str(row["video_path"]), manifest_file.parent),
            hidden_path=resolve_existing_path(str(row["hidden_path"]), manifest_file.parent),
        )
    return entries


def hidden_frame_indices(hidden_path: Path) -> np.ndarray:
    """Read the exact snippet frame indices emitted by the reusable hidden cache."""
    artifact = np.load(hidden_path, allow_pickle=False)
    if "frame_indices" not in artifact.files:
        raise ValueError(
            f"{hidden_path}: frame_indices are required for exact online CLIP alignment; "
            "re-extract or stage a manifest with this metadata"
        )
    indices = np.asarray(artifact["frame_indices"], dtype=np.int64).reshape(-1)
    if len(indices) == 0 or np.any(indices < 0) or np.any(np.diff(indices) <= 0):
        raise ValueError(f"{hidden_path}: invalid frame_indices")
    return indices


def decode_frames(video_path: Path, indices: np.ndarray) -> list[Image.Image]:
    """Decode exact indices with decord when available, otherwise OpenCV."""
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu(0))
        if int(indices[-1]) >= len(reader):
            raise ValueError(f"{video_path}: requested frame {indices[-1]}, video has {len(reader)} frames")
        # decord 0.6 returns one NDArray [T,H,W,3], not an iterable of
        # per-frame NDArrays.  Convert it once for compatibility with both
        # the extractor's batch semantics and current decord releases.
        batch = reader.get_batch(indices.tolist()).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in batch]
    except ImportError:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("install decord or opencv-python to decode online CLIP frames") from error
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")
        output = []
        try:
            for index in indices.tolist():
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"OpenCV cannot decode frame {index} from {video_path}")
                output.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        return output


class OnlineVideoDataset(Dataset):
    """Video frames aligned by the shared CLIP-hidden manifest, not guessed stride."""

    def __init__(
        self,
        source_csv: str,
        hidden_manifest: str,
        vadclip_root: str,
        skip_missing_manifest: bool = False,
    ) -> None:
        source_rows = read_csv(source_csv).reset_index(drop=True)
        self.entries = load_manifest(hidden_manifest)
        self.preprocess = build_clip_preprocess(vadclip_root)
        source_keys = source_rows["path"].map(lambda value: base_key(str(value)))
        absent = sorted(set(source_keys) - set(self.entries))
        self.missing_manifest_rows = source_rows.loc[source_keys.isin(absent), ["path", "label"]].copy()
        if absent:
            if not skip_missing_manifest:
                raise FileNotFoundError(
                    f"{len(absent)} source video keys ({len(self.missing_manifest_rows)} CSV rows) have no hidden-manifest entry; "
                    f"first={absent[0]!r}. Do not skip evaluation rows; training may opt in explicitly."
                )
            print(
                f"skip {len(self.missing_manifest_rows)} source rows from {len(absent)} videos without raw-video manifest entries",
                flush=True,
            )
            source_rows = source_rows.loc[~source_keys.isin(absent)]
        self.rows = source_rows.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> VideoSample:
        row = self.rows.loc[index]
        source_path, label = str(row["path"]), str(row["label"])
        entry = self.entries[base_key(source_path)]
        indices = hidden_frame_indices(entry.hidden_path)
        images = decode_frames(entry.video_path, indices)
        frames = torch.stack([self.preprocess(image) for image in images], dim=0)
        return VideoSample(frames=frames, label=label, source_path=source_path, video_path=str(entry.video_path))


def one_item_collate(items: list[VideoSample]) -> VideoSample:
    if len(items) != 1:
        raise ValueError("online CLIP loader uses one variable-length video at a time")
    return items[0]


def process_train_feature(feature: torch.Tensor, visual_length: int) -> tuple[torch.Tensor, int]:
    """Torch equivalent of the official ``utils.tools.process_feat``."""
    original_length = int(feature.shape[0])
    if original_length > visual_length:
        bounds = np.linspace(0, original_length, visual_length + 1, dtype=np.int32)
        pieces = [feature[left:right].mean(dim=0) if left != right else feature[left] for left, right in zip(bounds[:-1], bounds[1:])]
        return torch.stack(pieces, dim=0), visual_length
    if original_length < visual_length:
        feature = torch.nn.functional.pad(feature, (0, 0, 0, visual_length - original_length))
    return feature, original_length


def process_test_feature(feature: torch.Tensor, visual_length: int) -> tuple[torch.Tensor, int]:
    """Torch equivalent of official ``utils.tools.process_split``."""
    original_length = int(feature.shape[0])
    if original_length < visual_length:
        return torch.nn.functional.pad(feature, (0, 0, 0, visual_length - original_length)).unsqueeze(0), original_length
    chunks = []
    for index in range(int(original_length / visual_length) + 1):
        part = feature[index * visual_length:index * visual_length + visual_length]
        if part.shape[0] < visual_length:
            part = torch.nn.functional.pad(part, (0, 0, 0, visual_length - part.shape[0]))
        chunks.append(part)
    return torch.stack(chunks, dim=0), original_length


def test_chunk_lengths(original_length: int, visual_length: int) -> torch.Tensor:
    """Reproduce the official XD test chunk valid-length sequence exactly."""
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
    return torch.as_tensor(lengths, dtype=torch.int64)
