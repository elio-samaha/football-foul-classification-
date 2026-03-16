#!/usr/bin/env python3
"""Run VGGT-based 3D reconstruction on MVFoul clips.

Usage:
    # From CSV split (pre-extracted frames):
    python scripts/run_reconstruction.py --split_file configs/mvfoul_train.csv

    # Directly from raw MVFoul videos:
    python scripts/run_reconstruction.py --dataset_root data/mvfoul --split train

    # Using DPT fallback (no VGGT):
    python scripts/run_reconstruction.py --no-vggt --split_file configs/mvfoul_train.csv
"""
import argparse
from pathlib import Path

from src.vggt_reconstruction import ReconstructionConfig, ReconstructionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT reconstruction on MVFoul clips")
    parser.add_argument("--dataset_root", type=str, default="data/mvfoul")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", help="Split for video mode")
    parser.add_argument("--output_root", type=str, default="outputs/reconstruction")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--no-vggt", action="store_true", help="Use DPT fallback instead of VGGT")
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = ReconstructionConfig(
        dataset_root=Path(args.dataset_root),
        split_file=Path(args.split_file) if args.split_file else Path("configs/mvfoul_train.csv"),
        output_root=Path(args.output_root),
        model_name=args.model_name if not args.no_vggt else "Intel/dpt-hybrid-midas",
        use_vggt=not args.no_vggt,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
    )

    pipeline = ReconstructionPipeline(config)

    if args.split_file:
        pipeline.run_on_csv()
    else:
        pipeline.run_on_videos(args.split)


if __name__ == "__main__":
    main()
