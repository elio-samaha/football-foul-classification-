#!/usr/bin/env python3
"""Run VGGT-based 3D reconstruction on MVFoul actions.

Usage:
    python -m scripts.run_reconstruction --dataset-root data/mvfoul --split train --max-actions 5
    # or
    python scripts/run_reconstruction.py --dataset-root data/mvfoul --split train --max-actions 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vggt_reconstruction import ReconstructionConfig, ReconstructionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="VGGT reconstruction on MVFoul")
    parser.add_argument("--dataset-root", type=str, default="data/mvfoul", help="Path to MVFoul dataset root")
    parser.add_argument("--output-root", type=str, default="outputs/reconstruction", help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split: train, valid, test, challenge")
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B", help="VGGT model name")
    parser.add_argument("--max-actions", type=int, default=None, help="Limit number of actions to process")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames per clip")
    parser.add_argument("--frame-stride", type=int, default=4, help="Stride between sampled frames")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Point cloud confidence threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    config = ReconstructionConfig(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        split=args.split,
        vggt_model_name=args.model,
        num_frames_per_clip=args.num_frames,
        frame_stride=args.frame_stride,
        confidence_threshold=args.confidence_threshold,
        max_actions=args.max_actions,
        device=args.device,
    )
    ReconstructionPipeline(config).run()


if __name__ == "__main__":
    main()
