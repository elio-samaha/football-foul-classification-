from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class VGGTOutput:
    """Structured output from the VGGT model (or DPT fallback).

    Shapes assume B=1 for the common single-action case.
    """

    depth: torch.Tensor           # [B, S, H, W, 1]  or [B, 1, H, W] (DPT)
    depth_conf: Optional[torch.Tensor] = None    # [B, S, H, W]
    world_points: Optional[torch.Tensor] = None  # [B, S, H, W, 3]
    world_points_conf: Optional[torch.Tensor] = None  # [B, S, H, W]
    camera_poses: Optional[torch.Tensor] = None  # [B, S, 9]


class VGGTModel(nn.Module):
    """Wrapper around the real VGGT model (facebook/VGGT-1B).

    Accepts a batch of frames as [S, 3, H, W] or [B, S, 3, H, W]
    and returns depth, 3D world points, camera poses, and confidences.
    """

    def __init__(self, model_name: str = "facebook/VGGT-1B", dtype: str = "bfloat16") -> None:
        super().__init__()
        from vggt.models.vggt import VGGT

        self.model = VGGT.from_pretrained(model_name)
        self._dtype_str = dtype
        if dtype == "bfloat16":
            self._dtype = torch.bfloat16
        elif dtype == "float16":
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

    def forward(self, images: torch.Tensor) -> VGGTOutput:
        """Run VGGT inference.

        Args:
            images: [S, 3, H, W] or [B, S, 3, H, W], pixel values in [0, 1].
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)

        with torch.amp.autocast("cuda", dtype=self._dtype, enabled=images.is_cuda):
            preds = self.model(images)

        return VGGTOutput(
            depth=preds["depth"],
            depth_conf=preds.get("depth_conf"),
            world_points=preds.get("world_points"),
            world_points_conf=preds.get("world_points_conf"),
            camera_poses=preds.get("pose_enc"),
        )


class DPTDepthModel(nn.Module):
    """Fallback single-frame depth model using Intel DPT-Hybrid-MiDaS.

    Kept for environments where VGGT cannot run (no GPU, testing, etc.).
    """

    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas") -> None:
        super().__init__()
        try:
            from transformers import DPTForDepthEstimation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for DPT fallback. Install with: pip install transformers"
            ) from exc
        self.backbone = DPTForDepthEstimation.from_pretrained(model_name)

    def forward(self, pixel_values: torch.Tensor) -> VGGTOutput:
        """Run DPT depth estimation on a single frame.

        Args:
            pixel_values: [B, 3, H, W]
        """
        outputs = self.backbone(pixel_values=pixel_values)
        depth = outputs.predicted_depth.unsqueeze(1)  # [B, 1, H, W]
        confidence = torch.sigmoid(
            -torch.abs(depth - depth.mean(dim=(-1, -2), keepdim=True))
        )
        return VGGTOutput(depth=depth, depth_conf=confidence)


def load_model(
    model_name: str = "facebook/VGGT-1B",
    use_vggt: bool = True,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> nn.Module:
    """Load the appropriate model based on configuration."""
    if use_vggt:
        model = VGGTModel(model_name=model_name, dtype=dtype)
    else:
        model = DPTDepthModel(model_name=model_name)
    return model.to(device).eval()
