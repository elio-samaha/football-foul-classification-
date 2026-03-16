#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.vggt_reconstruction import (
    ReconstructionConfig,
    ReconstructionPipeline,
    download_mvfoul_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT-style reconstruction on SoccerNet MVFoul.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/SoccerNet/mvfouls"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/reconstruction"))
    parser.add_argument("--split-file", type=Path, default=Path("configs/reconstruction_split.csv"))
    parser.add_argument("--model-backend", choices=("vggt", "dpt"), default="vggt")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--resize-height", type=int, default=518)
    parser.add_argument("--resize-width", type=int, default=518)
    parser.add_argument("--resize-mode", choices=("pad", "crop", "stretch"), default="pad")
    parser.add_argument("--preferred-foul-frame", type=int, default=75)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--download-mvfoul", action="store_true")
    parser.add_argument("--soccernet-local-dir", type=Path, default=Path("data/SoccerNet"))
    parser.add_argument("--download-splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--download-version", type=str, default=None)
    parser.add_argument("--mvfoul-password", type=str, default=None)
    parser.add_argument("--no-save-depth-maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root

    if args.download_mvfoul:
        password = args.mvfoul_password or os.environ.get("SOCCERNET_MVFOUL_PASSWORD")
        if not password:
            raise ValueError(
                "MVFoul download requires a password. Pass --mvfoul-password or set SOCCERNET_MVFOUL_PASSWORD."
            )
        dataset_root = download_mvfoul_dataset(
            local_directory=args.soccernet_local_dir,
            password=password,
            splits=args.download_splits,
            version=args.download_version,
            extract=True,
        )

    default_model_name = "facebook/VGGT-1B" if args.model_backend == "vggt" else "Intel/dpt-hybrid-midas"
    config = ReconstructionConfig(
        dataset_root=dataset_root,
        output_root=args.output_root,
        split_file=args.split_file,
        model_backend=args.model_backend,
        model_name=args.model_name or default_model_name,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        resize_hw=(args.resize_height, args.resize_width),
        resize_mode=args.resize_mode,
        preferred_foul_frame=args.preferred_foul_frame,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        save_depth_maps=not args.no_save_depth_maps,
        max_clips=args.max_clips,
    )
    ReconstructionPipeline(config).run()


if __name__ == "__main__":
    main()
