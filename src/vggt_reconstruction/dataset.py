"""Dataset loader for SoccerNet MVFoul multi-view video clips.

The MVFoul dataset is organized as::

    <root>/
        Train/   (or Valid/, Test/, Chall/)
            action_1/
                clip_1.mp4
                clip_2.mp4
                ...
            action_2/
                ...

Each *action* folder contains multiple camera-view clips (mp4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_DIR_MAP = {
    "train": "Train",
    "valid": "Valid",
    "test": "Test",
    "challenge": "Chall",
}


@dataclass
class ActionSample:
    """One foul action with frames from multiple camera views."""

    action_id: str
    views: Dict[str, torch.Tensor]  # view_name -> [T, 3, H, W]
    view_names: List[str]
    annotations: Optional[dict] = None


@dataclass
class ClipSample:
    """A single clip (one camera view) of an action."""

    clip_id: str
    frames: torch.Tensor  # [T, 3, H, W]
    frame_paths: List[Path] = field(default_factory=list)
    labels: Optional[dict] = None


class MVFoulDataset(Dataset):
    """Loads multi-view clips from the SoccerNet MVFoul dataset.

    Each item yields an ``ActionSample`` with frames extracted from every
    available camera view for that action.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str = "train",
        num_frames: int = 8,
        frame_stride: int = 4,
        resize_hw: tuple[int, int] = (518, 518),
        max_actions: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.resize_hw = resize_hw

        split_dir_name = SPLIT_DIR_MAP.get(split, split)
        self.split_dir = self.dataset_root / split_dir_name
        if not self.split_dir.exists():
            alt = self.dataset_root / split
            if alt.exists():
                self.split_dir = alt
            else:
                raise FileNotFoundError(
                    f"Split directory not found: tried {self.split_dir} and {alt}. "
                    f"Available: {[p.name for p in self.dataset_root.iterdir() if p.is_dir()]}"
                )

        self.actions = self._discover_actions()
        if max_actions is not None:
            self.actions = self.actions[:max_actions]

        self.annotations = self._load_annotations()

    def _discover_actions(self) -> List[Path]:
        """Find all action folders (each containing >=1 clip mp4)."""
        actions = sorted(
            p for p in self.split_dir.iterdir()
            if p.is_dir() and any(p.glob("*.mp4"))
        )
        return actions

    def _load_annotations(self) -> dict:
        """Load annotation JSON if present alongside the split."""
        for name in ("annotations.json", "Labels-v2.json", "labels.json"):
            ann_path = self.split_dir / name
            if ann_path.exists():
                with ann_path.open() as f:
                    return json.load(f)

        parent_ann = self.dataset_root / "annotations.json"
        if parent_ann.exists():
            with parent_ann.open() as f:
                return json.load(f)
        return {}

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> ActionSample:
        action_dir = self.actions[idx]
        action_id = action_dir.name

        clips = sorted(action_dir.glob("*.mp4"))
        views: Dict[str, torch.Tensor] = {}
        view_names: List[str] = []

        for clip_path in clips:
            view_name = clip_path.stem
            frames = self._extract_frames(clip_path)
            if frames is not None:
                views[view_name] = frames
                view_names.append(view_name)

        annotations = None
        if self.annotations:
            annotations = self.annotations.get(action_id)

        return ActionSample(
            action_id=action_id,
            views=views,
            view_names=view_names,
            annotations=annotations,
        )

    def _extract_frames(self, video_path: Path) -> Optional[torch.Tensor]:
        """Extract temporally-strided frames from a video file."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None

        indices = self._sample_frame_indices(total_frames)
        frames: List[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = self.resize_hw
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            return None

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)  # [T, 3, H, W]
        return tensor

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """Compute frame indices centered around the middle of the clip."""
        center = total_frames // 2
        half_span = (self.num_frames * self.frame_stride) // 2
        start = max(0, center - half_span)

        indices = []
        for i in range(self.num_frames):
            idx = start + i * self.frame_stride
            idx = min(idx, total_frames - 1)
            indices.append(idx)
        return indices


class MVFoulFrameDataset(Dataset):
    """Flattened dataset: each item is one (action, view) pair as a ClipSample.

    Useful when you need a standard DataLoader-friendly dataset where each
    item is a single view clip.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str = "train",
        num_frames: int = 8,
        frame_stride: int = 4,
        resize_hw: tuple[int, int] = (518, 518),
        max_actions: int | None = None,
    ) -> None:
        self._base = MVFoulDataset(
            dataset_root=dataset_root,
            split=split,
            num_frames=num_frames,
            frame_stride=frame_stride,
            resize_hw=resize_hw,
            max_actions=max_actions,
        )
        self._index = self._build_flat_index()

    def _build_flat_index(self) -> List[tuple[int, Path]]:
        """Pre-scan all (action_idx, clip_path) pairs."""
        entries = []
        for action_idx, action_dir in enumerate(self._base.actions):
            for clip_path in sorted(action_dir.glob("*.mp4")):
                entries.append((action_idx, clip_path))
        return entries

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> ClipSample:
        action_idx, clip_path = self._index[idx]
        action_dir = self._base.actions[action_idx]
        action_id = action_dir.name
        clip_name = clip_path.stem
        clip_id = f"{action_id}/{clip_name}"

        frames = self._base._extract_frames(clip_path)
        if frames is None:
            frames = torch.zeros(self._base.num_frames, 3, *self._base.resize_hw)

        labels = None
        if self._base.annotations:
            labels = self._base.annotations.get(action_id)

        return ClipSample(
            clip_id=clip_id,
            frames=frames,
            labels=labels,
        )


SoccerNetMVFoulDataset = MVFoulFrameDataset
