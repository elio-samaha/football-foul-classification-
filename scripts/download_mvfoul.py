#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vggt_reconstruction import build_mvfoul_manifest, download_mvfoul_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SoccerNet MVFoul and build a reconstruction manifest.")
    parser.add_argument("--soccernet-local-dir", type=Path, default=Path("data/SoccerNet"))
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--manifest-path", type=Path, default=Path("configs/reconstruction_split.csv"))
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--preferred-foul-frame", type=int, default=75)
    parser.add_argument("--max-clips", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = args.password or os.environ.get("SOCCERNET_MVFOUL_PASSWORD")
    if not password:
        raise ValueError("Pass --password or set SOCCERNET_MVFOUL_PASSWORD before downloading MVFoul.")

    dataset_root = download_mvfoul_dataset(
        local_directory=args.soccernet_local_dir,
        password=password,
        splits=args.splits,
        version=args.version,
        extract=True,
    )
    manifest_path = build_mvfoul_manifest(
        dataset_root=dataset_root,
        output_path=args.manifest_path,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        preferred_foul_frame=args.preferred_foul_frame,
        max_clips=args.max_clips,
    )
    print(f"Downloaded MVFoul to {dataset_root}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
