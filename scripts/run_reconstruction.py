#!/usr/bin/env python3
from pathlib import Path

from src.vggt_reconstruction import ReconstructionConfig, ReconstructionPipeline


def main() -> None:
    config = ReconstructionConfig(
        dataset_root=Path("data/SoccerNet-MVFoul"),
        split_file=Path("configs/reconstruction_split.csv"),
        output_root=Path("outputs/reconstruction"),
        # Switch to the real VGGT backbone by setting backend="vggt" and
        # providing an appropriate model name from the facebookresearch/vggt
        # repository (e.g. "VGGT_Base" or "VGGT-1B-Commercial").
        backend="dpt",
        model_name="Intel/dpt-hybrid-midas",
    )
    ReconstructionPipeline(config).run()


if __name__ == "__main__":
    main()
