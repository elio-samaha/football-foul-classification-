import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import pytest

from src.vggt_reconstruction.dataset import (
    SoccerNetMVFoulDataset,
    ClipSample,
    MultiViewSample,
)


@pytest.fixture
def tmp_clip(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temporary clip with fake frames and a CSV split file."""
    frames_dir = tmp_path / "frames" / "clip_001"
    frames_dir.mkdir(parents=True)
    for i in range(10):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), img)

    csv_path = tmp_path / "split.csv"
    csv_path.write_text(f"clip_001,{frames_dir},0,10\n")
    return csv_path, frames_dir


def test_csv_dataset_loads(tmp_clip: tuple[Path, Path]) -> None:
    csv_path, _ = tmp_clip
    ds = SoccerNetMVFoulDataset(
        split_file=csv_path,
        sequence_length=4,
        frame_stride=2,
        resize_hw=(32, 32),
    )
    assert len(ds) == 1
    sample = ds[0]
    assert isinstance(sample, ClipSample)
    assert sample.frames.shape == (4, 3, 32, 32)
    assert sample.clip_id == "clip_001"


def test_multiview_sample() -> None:
    clip_a = ClipSample(clip_id="a", frames=torch.rand(4, 3, 32, 32), frame_paths=[])
    clip_b = ClipSample(clip_id="b", frames=torch.rand(4, 3, 32, 32), frame_paths=[])
    mv = MultiViewSample(action_id="act_001", clips=[clip_a, clip_b])
    assert mv.all_frames.shape == (8, 3, 32, 32)
