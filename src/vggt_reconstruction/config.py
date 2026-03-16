from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ReconstructionConfig:
    """Configuration for inference-time scene reconstruction."""

    dataset_root: Path
    output_root: Path
    split_file: Optional[Path] = None
    model_backend: str = "vggt"
    model_name: str = "facebook/VGGT-1B"
    sequence_length: int = 8
    frame_stride: int = 2
    resize_hw: Tuple[int, int] = (518, 518)
    resize_mode: str = "pad"
    preferred_foul_frame: int = 75
    confidence_threshold: float = 0.05
    device: str = "cuda"
    save_pointcloud_format: str = "ply"
    auto_build_manifest: bool = True
    save_depth_maps: bool = True
    max_clips: Optional[int] = None
    intrinsics_fx: float = 800.0
    intrinsics_fy: float = 800.0
    intrinsics_cx: float = 259.0
    intrinsics_cy: float = 259.0


@dataclass
class FinetuneConfig:
    """Configuration for optional fine-tuning of the depth backbone."""

    train_manifest: Path
    val_manifest: Path
    checkpoint_dir: Path
    model_backend: str = "dpt"
    init_model_name: str = "Intel/dpt-hybrid-midas"
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    sequence_length: int = 8
    frame_stride: int = 2
    resize_hw: Tuple[int, int] = (384, 384)
    resize_mode: str = "pad"
    mixed_precision: bool = True
    num_workers: int = 4
    max_grad_norm: Optional[float] = 1.0
    device: str = "cuda"
