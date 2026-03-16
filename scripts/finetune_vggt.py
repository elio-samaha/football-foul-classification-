#!/usr/bin/env python3
"""Fine-tune VGGT on MVFoul data.

Usage:
    python scripts/finetune_vggt.py --dataset-root data/mvfoul
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vggt_reconstruction.config import FinetuneConfig
from src.vggt_reconstruction.training import VGGTFinetuner


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune VGGT on MVFoul")
    parser.add_argument("--dataset-root", type=str, default="data/mvfoul", help="Path to MVFoul dataset root")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/vggt", help="Checkpoint output directory")
    parser.add_argument("--model", type=str, default="facebook/VGGT-1B", help="VGGT model name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    config = FinetuneConfig(
        dataset_root=Path(args.dataset_root),
        checkpoint_dir=Path(args.checkpoint_dir),
        vggt_model_name=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
    )
    VGGTFinetuner(config).fit()


if __name__ == "__main__":
    main()
