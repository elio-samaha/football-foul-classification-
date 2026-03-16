from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class ClipRecord:
    clip_id: str
    media_path: Path
    start_idx: int
    num_frames: int
    label_path: Optional[Path] = None


@dataclass
class ClipSample:
    clip_id: str
    frames: torch.Tensor  # [T, 3, H, W]
    frame_paths: List[Path]
    labels: Optional[dict] = None


def count_media_frames(media_path: Path) -> int:
    if media_path.is_dir():
        return len(list(_iter_image_paths(media_path)))
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise FileNotFoundError(media_path)
    try:
        return max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def build_mvfoul_manifest(
    dataset_root: Path,
    output_path: Path,
    sequence_length: int,
    frame_stride: int,
    preferred_foul_frame: int = 75,
    max_clips: Optional[int] = None,
) -> Path:
    """Create a manifest from a downloaded MVFoul folder tree.

    The downloader places split archives under `.../mvfouls/`. After extraction, this
    function discovers either raw videos or pre-extracted frame folders recursively.
    """

    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    media_paths = _discover_media_paths(dataset_root)
    if max_clips is not None:
        media_paths = media_paths[:max_clips]
    if not media_paths:
        raise FileNotFoundError(f"No MVFoul videos or frame folders found under {dataset_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "media_path", "start_idx", "num_frames", "label_path"])
        for media_path in media_paths:
            num_frames = count_media_frames(media_path)
            if num_frames <= 0:
                continue
            clip_span = 1 + max(0, sequence_length - 1) * frame_stride
            start_idx = _default_start_index(
                num_frames=num_frames,
                preferred_foul_frame=preferred_foul_frame,
                clip_span=clip_span,
            )
            label_path = _guess_label_path(media_path, dataset_root)
            clip_id = _clip_id_from_path(media_path, dataset_root)
            writer.writerow(
                [
                    clip_id,
                    str(media_path),
                    start_idx,
                    num_frames,
                    str(label_path) if label_path is not None else "",
                ]
            )
    return output_path


class SoccerNetMVFoulDataset(Dataset):
    """Loads temporally sampled MVFoul clips from either videos or frame folders."""

    def __init__(
        self,
        split_file: Path,
        sequence_length: int,
        frame_stride: int,
        resize_hw: tuple[int, int],
        resize_mode: str = "pad",
    ) -> None:
        self.samples = self._read_split_file(split_file)
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw
        self.resize_mode = resize_mode

    def _read_split_file(self, split_file: Path) -> List[ClipRecord]:
        raw_lines = [line for line in split_file.read_text().splitlines() if line.strip() and not line.startswith("#")]
        if not raw_lines:
            return []

        reader = csv.DictReader(raw_lines)
        if reader.fieldnames and "clip_id" in reader.fieldnames:
            return [self._record_from_mapping(row) for row in reader]

        rows: List[ClipRecord] = []
        for line in raw_lines:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid split row: {line}")
            label_path = Path(parts[4]) if len(parts) > 4 and parts[4] else None
            rows.append(
                ClipRecord(
                    clip_id=parts[0],
                    media_path=Path(parts[1]),
                    start_idx=int(parts[2]),
                    num_frames=int(parts[3]),
                    label_path=label_path,
                )
            )
        return rows

    @staticmethod
    def _record_from_mapping(row: dict) -> ClipRecord:
        media_path = row.get("media_path") or row.get("source_path") or row.get("frames_dir")
        if not media_path:
            raise ValueError(f"Manifest row is missing media path: {row}")
        label_value = row.get("label_path") or row.get("label_json_path") or ""
        return ClipRecord(
            clip_id=row["clip_id"],
            media_path=Path(media_path),
            start_idx=int(row["start_idx"]),
            num_frames=int(row["num_frames"]),
            label_path=Path(label_value) if label_value else None,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ClipSample:
        entry = self.samples[idx]
        frame_ids = [entry.start_idx + i * self.frame_stride for i in range(self.sequence_length)]
        frame_ids = [min(fid, max(0, entry.num_frames - 1)) for fid in frame_ids]
        if entry.media_path.is_dir():
            selected_paths, frames = self._load_from_frame_dir(entry.media_path, frame_ids)
        else:
            selected_paths, frames = self._load_from_video(entry.media_path, frame_ids)
        frame_tensor = torch.stack(frames, dim=0)
        return ClipSample(
            clip_id=entry.clip_id,
            frames=frame_tensor,
            frame_paths=selected_paths,
            labels=self._load_labels(entry.label_path),
        )

    def _load_from_frame_dir(self, frames_dir: Path, frame_ids: List[int]) -> tuple[List[Path], List[torch.Tensor]]:
        frame_paths = list(_iter_image_paths(frames_dir))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {frames_dir}")
        selected_paths = [frame_paths[min(fid, len(frame_paths) - 1)] for fid in frame_ids]
        return selected_paths, [self._load_image(path) for path in selected_paths]

    def _load_from_video(self, video_path: Path, frame_ids: List[int]) -> tuple[List[Path], List[torch.Tensor]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        frames: List[torch.Tensor] = []
        pseudo_paths: List[Path] = []
        last_valid_frame: Optional[np.ndarray] = None
        try:
            for frame_id in frame_ids:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, image = capture.read()
                if not ok:
                    if last_valid_frame is None:
                        raise RuntimeError(f"Could not decode frame {frame_id} from {video_path}")
                    image = last_valid_frame.copy()
                else:
                    last_valid_frame = image
                frames.append(self._image_to_tensor(image))
                pseudo_paths.append(video_path.with_name(f"{video_path.stem}_frame_{frame_id:04d}{video_path.suffix}"))
        finally:
            capture.release()
        return pseudo_paths, frames

    def _load_labels(self, label_path: Optional[Path]) -> Optional[dict]:
        if label_path is None or not label_path.exists():
            return None
        try:
            return json.loads(label_path.read_text())
        except json.JSONDecodeError:
            return {"source": str(label_path)}

    def _load_image(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        return self._image_to_tensor(image)

    def _image_to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = _resize_image(image, self.resize_hw, self.resize_mode)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


def _iter_image_paths(frames_dir: Path) -> Iterable[Path]:
    return sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def _discover_media_paths(dataset_root: Path) -> List[Path]:
    videos = sorted(path for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if videos:
        return videos

    frame_dirs: List[Path] = []
    for directory in sorted(path for path in dataset_root.rglob("*") if path.is_dir()):
        if any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in directory.iterdir()):
            frame_dirs.append(directory)
    return frame_dirs


def _clip_id_from_path(media_path: Path, dataset_root: Path) -> str:
    relative_path = media_path.relative_to(dataset_root)
    return relative_path.with_suffix("").as_posix() if media_path.is_file() else relative_path.as_posix()


def _guess_label_path(media_path: Path, dataset_root: Path) -> Optional[Path]:
    stem = media_path.stem if media_path.is_file() else media_path.name
    candidates = [
        media_path.with_suffix(".json") if media_path.is_file() else media_path / "labels.json",
        media_path.parent / f"{stem}.json",
        dataset_root / "labels" / f"{stem}.json",
        dataset_root / "annotations" / f"{stem}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _default_start_index(num_frames: int, preferred_foul_frame: int, clip_span: int) -> int:
    latest_valid_start = max(0, num_frames - clip_span)
    centered_start = max(0, preferred_foul_frame - clip_span // 2)
    return min(centered_start, latest_valid_start)


def _resize_image(image: np.ndarray, resize_hw: tuple[int, int], resize_mode: str) -> np.ndarray:
    target_h, target_w = resize_hw
    source_h, source_w = image.shape[:2]

    if resize_mode == "stretch":
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if resize_mode not in {"pad", "crop"}:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    if resize_mode == "pad":
        scale = min(target_w / source_w, target_h / source_h)
    else:
        scale = max(target_w / source_w, target_h / source_h)

    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    if resize_mode == "crop":
        start_y = max(0, (resized_h - target_h) // 2)
        start_x = max(0, (resized_w - target_w) // 2)
        return resized[start_y : start_y + target_h, start_x : start_x + target_w]

    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    offset_y = (target_h - resized_h) // 2
    offset_x = (target_w - resized_w) // 2
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    return canvas
