#!/usr/bin/env python3
"""Extract frames from MVFoul videos and generate split CSV manifests.

Expected input layout (after unzipping the SoccerNet download):
    data/mvfoul/Train/action_XXXX/clip_N.mp4
    data/mvfoul/Valid/action_XXXX/clip_N.mp4
    data/mvfoul/Test/action_XXXX/clip_N.mp4

Produces:
    data/mvfoul/frames/{split}/action_XXXX/clip_N/frame_NNNN.jpg
    configs/mvfoul_train.csv
    configs/mvfoul_valid.csv
    configs/mvfoul_test.csv

Usage:
    python scripts/prepare_mvfoul.py --dataset_dir data/mvfoul [--fps 5] [--max_actions 0]
"""
import argparse
import json
from pathlib import Path

import cv2


SPLIT_FOLDER_MAP = {
    "train": "Train",
    "valid": "Valid",
    "test": "Test",
    "challenge": "Chall",
}


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 5.0,
) -> int:
    """Extract frames from a video at the given target FPS. Returns frame count."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    frame_interval = max(1, round(src_fps / target_fps))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def find_annotation_file(dataset_dir: Path) -> dict:
    """Try to locate and load the annotations JSON for the dataset."""
    candidates = list(dataset_dir.rglob("annotations*.json")) + list(
        dataset_dir.rglob("Actions*.json")
    )
    if not candidates:
        return {}
    ann_path = candidates[0]
    print(f"Found annotations: {ann_path}")
    with open(ann_path) as f:
        return json.load(f)


def process_split(
    dataset_dir: Path,
    split_name: str,
    frames_root: Path,
    target_fps: float,
    max_actions: int,
) -> list[dict]:
    """Process one split, return rows for the CSV manifest."""
    folder_name = SPLIT_FOLDER_MAP.get(split_name, split_name.capitalize())
    split_dir = dataset_dir / folder_name

    if not split_dir.exists():
        print(f"Split directory not found: {split_dir}, skipping.")
        return []

    action_dirs = sorted(
        [d for d in split_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if max_actions > 0:
        action_dirs = action_dirs[:max_actions]

    rows = []
    for action_dir in action_dirs:
        action_id = action_dir.name
        videos = sorted(action_dir.glob("*.mp4"))
        if not videos:
            videos = sorted(action_dir.glob("*.avi"))
        if not videos:
            continue

        for video_path in videos:
            clip_name = video_path.stem
            clip_id = f"{split_name}_{action_id}_{clip_name}"
            out_dir = frames_root / split_name / action_id / clip_name
            n_frames = extract_frames(video_path, out_dir, target_fps)
            if n_frames == 0:
                continue
            rows.append(
                {
                    "clip_id": clip_id,
                    "frames_dir": str(out_dir),
                    "start_idx": 0,
                    "num_frames": n_frames,
                    "action_id": action_id,
                    "split": split_name,
                }
            )

    print(f"  {split_name}: {len(rows)} clips from {len(action_dirs)} actions")
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# clip_id,frames_dir,start_idx,num_frames\n")
        for r in rows:
            f.write(f"{r['clip_id']},{r['frames_dir']},{r['start_idx']},{r['num_frames']}\n")
    print(f"  Wrote {len(rows)} entries to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MVFoul dataset: extract frames and generate splits")
    parser.add_argument("--dataset_dir", type=str, default="data/mvfoul", help="Root of the MVFoul dataset")
    parser.add_argument("--frames_dir", type=str, default=None, help="Where to write extracted frames (default: {dataset_dir}/frames)")
    parser.add_argument("--configs_dir", type=str, default="configs", help="Where to write CSV manifests")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for frame extraction")
    parser.add_argument("--max_actions", type=int, default=0, help="Max actions per split (0 = all)")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Splits to process")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    frames_dir = Path(args.frames_dir) if args.frames_dir else dataset_dir / "frames"
    configs_dir = Path(args.configs_dir)

    print(f"Dataset root: {dataset_dir}")
    print(f"Frames output: {frames_dir}")
    print(f"Target FPS: {args.fps}")

    for split in args.splits:
        rows = process_split(dataset_dir, split, frames_dir, args.fps, args.max_actions)
        if rows:
            write_csv(rows, configs_dir / f"mvfoul_{split}.csv")

    print("\nDone! You can now run the reconstruction pipeline with the generated CSVs.")


if __name__ == "__main__":
    main()
