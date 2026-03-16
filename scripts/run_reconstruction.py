#!/usr/bin/env python3
from pathlib import Path

from src.vggt_reconstruction import ReconstructionConfig, ReconstructionPipeline


def main() -> None:
    config = ReconstructionConfig(
        dataset_root=Path("data/SoccerNet-MVFoul"),
        split_file=Path("configs/reconstruction_split.csv"),
        output_root=Path("outputs/reconstruction"),
    )
    ReconstructionPipeline(config).run()


if __name__ == "__main__":
    main()
