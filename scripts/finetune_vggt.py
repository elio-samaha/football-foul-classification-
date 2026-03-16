#!/usr/bin/env python3
from pathlib import Path

from src.vggt_reconstruction import FinetuneConfig
from src.vggt_reconstruction.training import VGGTFinetuner


def main() -> None:
    config = FinetuneConfig(
        train_manifest=Path("configs/train_split.csv"),
        val_manifest=Path("configs/val_split.csv"),
        checkpoint_dir=Path("checkpoints/vggt_depth"),
    )
    VGGTFinetuner(config).fit()


if __name__ == "__main__":
    main()
