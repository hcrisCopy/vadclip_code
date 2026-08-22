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

from common import add_vadclip_source, base_key, load_feature, read_csv, resolve_existing_path


@dataclass(frozen=True)
class VideoSample:
    """One source feature row and its exact raw frames used by hidden extraction."""

    frames: torch.Tensor
    source_feature: torch.Tensor
    label: str
    source_path: str
    video_path: str
    crop_type: int


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


def source_crop_type(source_path: str) -> int:
    """Read the original XD 10-crop code from a ``...__0.npy`` feature name."""
    stem = Path(source_path).stem
    _head, marker, tail = stem.rpartition("__")
    if not marker or not tail.isdigit() or not 0 <= int(tail) <= 9:
        raise ValueError(f"{source_path}: expected XD feature name ending in __0 through __9")
    return int(tail)


def vadclip_xd_crop(frame_bgr: np.ndarray, crop_type: int) -> Image.Image:
    """Independent reproduction of official ``VadCLIP/src/crop.py:image_crop``."""
    import cv2

    image = cv2.resize(frame_bgr, dsize=(340, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if crop_type == 0:
        image = image[16:240, 58:282, :]
    elif crop_type == 1:
        image = image[:224, :224, :]
    elif crop_type == 2:
        image = image[:224, -224:, :]
    elif crop_type == 3:
        image = image[-224:, :224, :]
    elif crop_type == 4:
        image = image[-224:, -224:, :]
    elif crop_type == 5:
        image = cv2.flip(image[16:240, 58:282, :], 1)
    elif crop_type == 6:
        image = cv2.flip(image[:224, :224, :], 1)
    elif crop_type == 7:
        image = cv2.flip(image[:224, -224:, :], 1)
    elif crop_type == 8:
        image = cv2.flip(image[-224:, :224, :], 1)
    elif crop_type == 9:
        image = cv2.flip(image[-224:, -224:, :], 1)
    else:
        raise ValueError(f"crop_type must be in [0,9], got {crop_type}")
    return Image.fromarray(image).convert("RGB")


def decode_frames(video_path: Path, indices: np.ndarray, crop_type: int) -> list[Image.Image]:
    """Decode and spatially crop exact frames as the original XD feature cache did."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("opencv-python is required to reproduce XD's original 10-crop features") from error
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")
    wanted = {int(frame): position for position, frame in enumerate(indices.tolist())}
    output: list[Image.Image | None] = [None] * len(indices)
    frame_index = 0
    try:
        while frame_index <= int(indices[-1]):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"OpenCV cannot decode frame {frame_index} from {video_path}")
            position = wanted.get(frame_index)
            if position is not None:
                output[position] = vadclip_xd_crop(frame, crop_type)
            frame_index += 1
    finally:
        capture.release()
    if any(frame is None for frame in output):
        raise RuntimeError(f"{video_path}: unable to decode every requested frame")
    return [frame for frame in output if frame is not None]


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
        crop_type = source_crop_type(source_path)
        entry = self.entries[base_key(source_path)]
        indices = hidden_frame_indices(entry.hidden_path)
        # The established concat builder treats the old 512D source feature
        # length as the temporal contract and crops a longer hidden sequence
        # at its tail.  Do the same before decoding online frames.  We read
        # Adapter training keeps this as an immutable VadCLIP feature anchor;
        # only the online CLIP difference induced by the Adapter is added.
        source_feature = load_feature(resolve_existing_path(source_path))
        target_length = int(source_feature.shape[0])
        if len(indices) < target_length:
            raise ValueError(
                f"{source_path}: hidden manifest has {len(indices)} snippets but source feature requires {target_length}"
            )
        indices = indices[:target_length]
        images = decode_frames(entry.video_path, indices, crop_type)
        frames = torch.stack([self.preprocess(image) for image in images], dim=0)
        return VideoSample(
            frames=frames, source_feature=torch.from_numpy(source_feature), label=label, source_path=source_path,
            video_path=str(entry.video_path), crop_type=crop_type,
        )


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
