#!/usr/bin/env python3
"""
Prepare MVFoul frame folders and split CSVs from the raw SoccerNet videos.

The official SoccerNet API downloads MVFoul as multi-view videos organised in
Train / Valid / Test / Chall folders. For the VGGT reconstruction pipeline in
this repo we work with per-clip RGB frame folders configured through a CSV:

    clip_id,frames_dir,start_idx,num_frames,label_json_path

This script:
1. Scans the downloaded MVFoul videos.
2. Extracts a configurable number of frames per video with OpenCV.
3. Writes frame folders under `data/SoccerNet-MVFoul/frames/{clip_id}`.
4. Emits a CSV manifest compatible with `SoccerNetMVFoulDataset`.

Example
-------
Assuming you have run `scripts/download_mvfoul.py` with default arguments:

    python scripts/prepare_mvfoul_frames.py \\
        --soccernet-root data/SoccerNet \\
        --output-root data/SoccerNet-MVFoul \\
        --split Train \\
        --max-videos 50

This will create:
- frame folders in `data/SoccerNet-MVFoul/frames/`
- a CSV manifest at `configs/reconstruction_split.csv` (by default)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2


def iter_video_files(root: Path, split: str) -> Iterable[Path]:
    """Yield all video files under the given SoccerNet root/split."""
    split_root = root / "mvfouls" / split
    if not split_root.exists():
        raise FileNotFoundError(
            f"Expected split folder {split_root} does not exist. "
            "Make sure you downloaded MVFoul with the SoccerNet API."
        )
    for ext in ("*.mkv", "*.mp4", "*.avi"):
        yield from split_root.rglob(ext)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    max_frames: int,
    resize_hw: Tuple[int, int],
) -> int:
    """Extract up to `max_frames` frames from a video into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    h, w = resize_hw
    frame_count = 0
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_idx = frame_count
        out_path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(out_path), frame)
        frame_count += 1

    cap.release()
    return frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MVFoul frame folders and CSV manifest.")
    parser.add_argument(
        "--soccernet-root",
        type=Path,
        default=Path("data/SoccerNet"),
        help="Root directory passed as LocalDirectory to SoccerNetDownloader.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/SoccerNet-MVFoul"),
        help="Root directory where frame folders and labels will be written.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Train",
        choices=["Train", "Valid", "Test", "Chall", "TrainMini"],
        help="MVFoul split to convert.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=100,
        help="Maximum number of videos to convert (useful for quick experiments).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=96,
        help="Maximum number of frames to extract per video.",
    )
    parser.add_argument(
        "--resize-h",
        type=int,
        default=384,
        help="Resized frame height.",
    )
    parser.add_argument(
        "--resize-w",
        type=int,
        default=384,
        help="Resized frame width.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("configs/reconstruction_split.csv"),
        help="Path to CSV manifest to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_paths: List[Path] = list(iter_video_files(args.soccernet_root, args.split))
    if not video_paths:
        raise SystemExit(
            f"No videos found under {args.soccernet_root}/mvfouls/{args.split}. "
            "Verify that the dataset is downloaded and the split name is correct."
        )

    video_paths = sorted(video_paths)[: args.max_videos]

    output_frames_root = args.output_root / "frames"
    labels_root = args.output_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, Path, int, int, Path]] = []

    for idx, video_path in enumerate(video_paths):
        clip_id = video_path.stem
        clip_frames_dir = output_frames_root / clip_id
        num_frames = extract_frames(
            video_path=video_path,
            out_dir=clip_frames_dir,
            max_frames=args.max_frames_per_video,
            resize_hw=(args.resize_h, args.resize_w),
        )
        # Placeholder label path – you can later map MVFoul annotations here.
        label_path = labels_root / f"{clip_id}.json"
        rows.append((clip_id, clip_frames_dir, 0, num_frames, label_path))
        print(f"[{idx+1}/{len(video_paths)}] {video_path} -> {clip_frames_dir} ({num_frames} frames)")

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_id", "frames_dir", "start_idx", "num_frames", "label_json_path"])
        for clip_id, frames_dir, start_idx, num_frames, label_path in rows:
            writer.writerow([clip_id, frames_dir, start_idx, num_frames, label_path])

    print(f"Wrote manifest with {len(rows)} clips to {args.manifest_path}")


if __name__ == "__main__":
    main()

