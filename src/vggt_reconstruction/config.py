from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ReconstructionConfig:
    """Configuration for inference-time scene reconstruction using VGGT."""

    dataset_root: Path
    split_file: Path
    output_root: Path

    model_name: str = "facebook/VGGT-1B"
    use_vggt: bool = True

    sequence_length: int = 8
    frame_stride: int = 2
    resize_hw: Tuple[int, int] = (518, 518)
    confidence_threshold: float = 0.5
    device: str = "cuda"
    dtype: str = "bfloat16"
    save_pointcloud_format: str = "ply"

    # Fallback intrinsics (used only when VGGT is disabled)
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

    init_model_name: str = "facebook/VGGT-1B"
    use_vggt: bool = True

    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    sequence_length: int = 8
    frame_stride: int = 2
    resize_hw: Tuple[int, int] = (518, 518)
    mixed_precision: bool = True
    num_workers: int = 4
    max_grad_norm: Optional[float] = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class MVFoulAction:
    """Metadata for a single action in the MVFoul dataset."""

    action_id: str
    split: str
    clip_paths: list = field(default_factory=list)
    offence_severity: Optional[str] = None
    action_class: Optional[str] = None

    OFFENCE_LABELS = [
        "No offence",
        "Offence + No card",
        "Offence + Yellow card",
        "Offence + Red card",
    ]

    ACTION_LABELS = [
        "Standing tackling",
        "Tackling",
        "Holding",
        "Pushing",
        "Challenge",
        "Dive",
        "High leg",
        "Elbowing",
    ]
