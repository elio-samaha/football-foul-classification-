from pathlib import Path

import cv2
import numpy as np

from src.vggt_reconstruction.dataset import SoccerNetMVFoulDataset, build_mvfoul_manifest


def _write_dummy_frame(path: Path, value: int) -> None:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_build_manifest_and_load_frame_directory(tmp_path: Path) -> None:
    clip_dir = tmp_path / "mvfouls" / "Train" / "clip_0001"
    clip_dir.mkdir(parents=True)
    for idx in range(6):
        _write_dummy_frame(clip_dir / f"frame_{idx:04d}.jpg", value=idx * 10)

    manifest_path = tmp_path / "manifest.csv"
    build_mvfoul_manifest(
        dataset_root=tmp_path / "mvfouls",
        output_path=manifest_path,
        sequence_length=4,
        frame_stride=1,
    )

    dataset = SoccerNetMVFoulDataset(
        split_file=manifest_path,
        sequence_length=4,
        frame_stride=1,
        resize_hw=(32, 32),
        resize_mode="pad",
    )

    sample = dataset[0]
    assert sample.clip_id == "Train/clip_0001"
    assert sample.frames.shape == (4, 3, 32, 32)
    assert len(sample.frame_paths) == 4
