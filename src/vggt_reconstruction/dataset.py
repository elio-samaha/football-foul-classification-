from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ClipSample:
    """A single video clip sample (one camera view of one action)."""

    clip_id: str
    frames: torch.Tensor  # [T, 3, H, W]
    frame_paths: List[Path]
    labels: Optional[dict] = None


@dataclass
class MultiViewSample:
    """All camera views for a single action, suitable for VGGT multi-view input."""

    action_id: str
    clips: List[ClipSample] = field(default_factory=list)
    labels: Optional[dict] = None

    @property
    def all_frames(self) -> torch.Tensor:
        """Stack all view frames into [total_views * T, 3, H, W]."""
        return torch.cat([c.frames for c in self.clips], dim=0)


class SoccerNetMVFoulDataset(Dataset):
    """Loads clips from SoccerNet MVFoul frame folders.

    Supports two modes:
    1. CSV manifest mode: reads a CSV with pre-extracted frame paths.
    2. Video directory mode: reads .mp4 videos directly and extracts frames.

    CSV format:
        clip_id,frames_dir,start_idx,num_frames[,label_json_path]
    """

    def __init__(
        self,
        split_file: Path,
        sequence_length: int,
        frame_stride: int,
        resize_hw: tuple[int, int],
    ) -> None:
        self.samples = self._read_split_file(split_file)
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw

    def _read_split_file(self, split_file: Path) -> List[dict]:
        rows: List[dict] = []
        for line in split_file.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid split row: {line}")
            rows.append(
                {
                    "clip_id": parts[0],
                    "frames_dir": Path(parts[1]),
                    "start_idx": int(parts[2]),
                    "num_frames": int(parts[3]),
                    "label_path": Path(parts[4]) if len(parts) > 4 and parts[4] else None,
                }
            )
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ClipSample:
        entry = self.samples[idx]
        frame_paths = sorted(entry["frames_dir"].glob("*.jpg"))
        if not frame_paths:
            frame_paths = sorted(entry["frames_dir"].glob("*.png"))
        start = entry["start_idx"]
        frame_ids = [start + i * self.frame_stride for i in range(self.sequence_length)]
        frame_ids = [min(fid, len(frame_paths) - 1) for fid in frame_ids]
        selected_paths = [frame_paths[fid] for fid in frame_ids]
        frames = [self._load_image(path) for path in selected_paths]
        frame_tensor = torch.stack(frames, dim=0)
        labels = None
        if entry["label_path"] and entry["label_path"].exists():
            labels = {"source": str(entry["label_path"])}
        return ClipSample(
            clip_id=entry["clip_id"],
            frames=frame_tensor,
            frame_paths=selected_paths,
            labels=labels,
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = self.resize_hw
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)


class MVFoulVideoDataset(Dataset):
    """Loads MVFoul actions directly from the raw video directory structure.

    Expected layout:
        dataset_root/{Train,Valid,Test}/action_XXXX/clip_N.mp4

    Each item returns a MultiViewSample with frames from all camera views.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        sequence_length: int = 8,
        frame_stride: int = 2,
        resize_hw: tuple[int, int] = (518, 518),
        max_clips_per_action: int = 4,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw
        self.max_clips_per_action = max_clips_per_action

        split_folder_map = {
            "train": "Train",
            "valid": "Valid",
            "test": "Test",
            "challenge": "Chall",
        }
        self.split_dir = self.dataset_root / split_folder_map.get(split, split.capitalize())
        self.action_dirs = self._discover_actions()

    def _discover_actions(self) -> List[Path]:
        if not self.split_dir.exists():
            return []
        return sorted(
            [d for d in self.split_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

    def __len__(self) -> int:
        return len(self.action_dirs)

    def __getitem__(self, idx: int) -> MultiViewSample:
        action_dir = self.action_dirs[idx]
        action_id = action_dir.name

        videos = sorted(action_dir.glob("*.mp4"))
        if not videos:
            videos = sorted(action_dir.glob("*.avi"))

        clips: List[ClipSample] = []
        for vid_path in videos[: self.max_clips_per_action]:
            frames, paths = self._decode_video(vid_path)
            if frames is not None:
                clip_id = f"{self.split}_{action_id}_{vid_path.stem}"
                clips.append(ClipSample(clip_id=clip_id, frames=frames, frame_paths=paths))

        return MultiViewSample(action_id=action_id, clips=clips)

    def _decode_video(self, video_path: Path) -> tuple:
        """Extract frames from a video file using OpenCV."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 25.0

        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        if not all_frames:
            return None, []

        n_available = len(all_frames)
        center = n_available // 2
        half_span = (self.sequence_length * self.frame_stride) // 2
        start = max(0, center - half_span)

        indices = [start + i * self.frame_stride for i in range(self.sequence_length)]
        indices = [min(idx, n_available - 1) for idx in indices]

        h, w = self.resize_hw
        tensors = []
        for idx in indices:
            bgr = all_frames[idx]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            tensors.append(tensor)

        paths = [video_path] * len(indices)
        return torch.stack(tensors, dim=0), paths
