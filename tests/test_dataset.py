"""Tests for the MVFoul dataset loader.

These tests create a tiny synthetic dataset to verify the loading logic
without needing the real MVFoul data.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from src.vggt_reconstruction.dataset import (
    MVFoulDataset,
    MVFoulFrameDataset,
    ActionSample,
    ClipSample,
)


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a minimal fake MVFoul directory tree with tiny mp4 clips."""
    train_dir = tmp_path / "Train"
    for action_idx in range(3):
        action_dir = train_dir / f"action_{action_idx}"
        action_dir.mkdir(parents=True)
        for clip_idx in range(2):
            clip_path = action_dir / f"clip_{clip_idx}.mp4"
            _create_tiny_video(clip_path, n_frames=20, h=64, w=64)

    valid_dir = tmp_path / "Valid"
    action_dir = valid_dir / "action_99"
    action_dir.mkdir(parents=True)
    _create_tiny_video(action_dir / "clip_0.mp4", n_frames=10, h=64, w=64)

    return tmp_path


def _create_tiny_video(path: Path, n_frames: int, h: int, w: int) -> None:
    """Generate a tiny mp4 video with random frames using OpenCV."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25, (w, h))
    for _ in range(n_frames):
        frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestMVFoulDataset:
    def test_discover_actions(self, synthetic_dataset: Path) -> None:
        ds = MVFoulDataset(synthetic_dataset, split="train", num_frames=4, frame_stride=2)
        assert len(ds) == 3

    def test_getitem_returns_action_sample(self, synthetic_dataset: Path) -> None:
        ds = MVFoulDataset(
            synthetic_dataset, split="train", num_frames=4, frame_stride=2,
            resize_hw=(32, 32),
        )
        sample = ds[0]
        assert isinstance(sample, ActionSample)
        assert len(sample.views) == 2
        for view_tensor in sample.views.values():
            assert view_tensor.shape == (4, 3, 32, 32)

    def test_max_actions(self, synthetic_dataset: Path) -> None:
        ds = MVFoulDataset(
            synthetic_dataset, split="train", num_frames=4, frame_stride=2,
            max_actions=1,
        )
        assert len(ds) == 1

    def test_valid_split(self, synthetic_dataset: Path) -> None:
        ds = MVFoulDataset(synthetic_dataset, split="valid", num_frames=4, frame_stride=2)
        assert len(ds) == 1


class TestMVFoulFrameDataset:
    def test_flat_index(self, synthetic_dataset: Path) -> None:
        ds = MVFoulFrameDataset(
            synthetic_dataset, split="train", num_frames=4, frame_stride=2,
        )
        assert len(ds) == 6  # 3 actions * 2 clips

    def test_getitem_returns_clip_sample(self, synthetic_dataset: Path) -> None:
        ds = MVFoulFrameDataset(
            synthetic_dataset, split="train", num_frames=4, frame_stride=2,
            resize_hw=(32, 32),
        )
        sample = ds[0]
        assert isinstance(sample, ClipSample)
        assert sample.frames.shape == (4, 3, 32, 32)
        assert "/" in sample.clip_id
