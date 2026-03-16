#!/usr/bin/env python3
"""Fine-tune the VGGT (or DPT) depth model on MVFoul frames with depth supervision.

Usage:
    python scripts/finetune_vggt.py \
        --train_manifest configs/mvfoul_train.csv \
        --val_manifest configs/mvfoul_valid.csv

    # With DPT fallback:
    python scripts/finetune_vggt.py --no-vggt
"""
import argparse
from pathlib import Path

from src.vggt_reconstruction import FinetuneConfig
from src.vggt_reconstruction.training import VGGTFinetuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune depth model on MVFoul")
    parser.add_argument("--train_manifest", type=str, default="configs/mvfoul_train.csv")
    parser.add_argument("--val_manifest", type=str, default="configs/mvfoul_valid.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vggt_depth")
    parser.add_argument("--model_name", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--no-vggt", action="store_true", help="Use DPT fallback instead of VGGT")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = FinetuneConfig(
        train_manifest=Path(args.train_manifest),
        val_manifest=Path(args.val_manifest),
        checkpoint_dir=Path(args.checkpoint_dir),
        init_model_name=args.model_name if not args.no_vggt else "Intel/dpt-hybrid-midas",
        use_vggt=not args.no_vggt,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    VGGTFinetuner(config).fit()


if __name__ == "__main__":
    main()
