"""VGGT model wrapper for 3D reconstruction on soccer footage.

Uses the official facebook/VGGT-1B model from Meta AI Research
(CVPR 2025 Best Paper).  The model predicts depth maps, 3D world
point maps, camera pose encodings, and confidence scores from one or
more image views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    _VGGT_AVAILABLE = True
except ImportError:
    _VGGT_AVAILABLE = False


@dataclass
class VGGTOutput:
    """Structured output from a VGGT forward pass."""

    depth: torch.Tensor              # [B, S, H, W, 1]
    depth_conf: torch.Tensor         # [B, S, H, W]
    world_points: torch.Tensor       # [B, S, H, W, 3]
    world_points_conf: torch.Tensor  # [B, S, H, W]
    pose_enc: torch.Tensor           # [B, S, 9]
    extrinsics: Optional[torch.Tensor] = None  # [B, S, 3, 4]
    intrinsics: Optional[torch.Tensor] = None  # [B, S, 3, 3]
    raw: Dict[str, torch.Tensor] = field(default_factory=dict)


class VGGTModel(nn.Module):
    """Wrapper around the official VGGT-1B model."""

    def __init__(
        self,
        model_name: str = "facebook/VGGT-1B",
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        if not _VGGT_AVAILABLE:
            raise ImportError(
                "VGGT is not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/vggt.git"
            )

        self._dtype_str = dtype
        self._torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        self.model = VGGT.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> VGGTOutput:
        """Run VGGT inference.

        Args:
            images: Tensor of shape ``[S, 3, H, W]`` or ``[B, S, 3, H, W]``
                    with values in ``[0, 1]``.

        Returns:
            VGGTOutput with depth, point maps, camera params, and confidence.
        """
        with torch.cuda.amp.autocast(dtype=self._torch_dtype):
            predictions = self.model(images)

        h, w = images.shape[-2], images.shape[-1]

        extrinsics, intrinsics = None, None
        if "pose_enc" in predictions:
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"], image_size_hw=(h, w)
            )

        return VGGTOutput(
            depth=predictions["depth"],
            depth_conf=predictions["depth_conf"],
            world_points=predictions["world_points"],
            world_points_conf=predictions["world_points_conf"],
            pose_enc=predictions["pose_enc"],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            raw=predictions,
        )


VGGTDepthModel = VGGTModel
