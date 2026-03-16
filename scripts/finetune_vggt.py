#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.vggt_reconstruction import FinetuneConfig
from src.vggt_reconstruction.training import VGGTFinetuner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the depth branch on MVFoul manifests.")
    parser.add_argument("--train-manifest", type=Path, default=Path("configs/train_split.csv"))
    parser.add_argument("--val-manifest", type=Path, default=Path("configs/val_split.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/vggt_depth"))
    parser.add_argument("--model-backend", choices=("vggt", "dpt"), default="dpt")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--resize-height", type=int, default=384)
    parser.add_argument("--resize-width", type=int, default=384)
    parser.add_argument("--resize-mode", choices=("pad", "crop", "stretch"), default="pad")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_model_name = "facebook/VGGT-1B" if args.model_backend == "vggt" else "Intel/dpt-hybrid-midas"
    config = FinetuneConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        checkpoint_dir=args.checkpoint_dir,
        model_backend=args.model_backend,
        init_model_name=args.model_name or default_model_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sequence_length=args.sequence_length,
        frame_stride=args.frame_stride,
        resize_hw=(args.resize_height, args.resize_width),
        resize_mode=args.resize_mode,
        device=args.device,
    )
    VGGTFinetuner(config).fit()


if __name__ == "__main__":
    main()
