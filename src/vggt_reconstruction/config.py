from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ReconstructionConfig:
    """Configuration for inference-time scene reconstruction using VGGT."""

    dataset_root: Path
    output_root: Path
    split: str = "train"
    vggt_model_name: str = "facebook/VGGT-1B"
    num_views: int = 2
    num_frames_per_clip: int = 8
    frame_stride: int = 4
    resize_hw: Tuple[int, int] = (518, 518)
    confidence_threshold: float = 0.5
    device: str = "cuda"
    dtype: str = "bfloat16"
    save_pointcloud_format: str = "ply"
    max_actions: Optional[int] = None


@dataclass
class FinetuneConfig:
    """Configuration for optional fine-tuning of the depth backbone."""

    dataset_root: Path
    checkpoint_dir: Path
    vggt_model_name: str = "facebook/VGGT-1B"
    train_split: str = "train"
    val_split: str = "valid"
    epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    num_frames_per_clip: int = 4
    frame_stride: int = 8
    resize_hw: Tuple[int, int] = (518, 518)
    mixed_precision: bool = True
    num_workers: int = 2
    max_grad_norm: Optional[float] = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"
