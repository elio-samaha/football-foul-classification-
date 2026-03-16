from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class ClipSample:
    clip_id: str
    frames: torch.Tensor  # [T, 3, H, W]
    frame_paths: List[Path]
    labels: Optional[dict] = None


class SoccerNetMVFoulDataset(Dataset):
    """Loads temporally sampled clips from SoccerNet MVFoul frame folders.

    Expected split file format (CSV):
        clip_id,frames_dir,start_idx,num_frames,label_json_path(optional)
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
