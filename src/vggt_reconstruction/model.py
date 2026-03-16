from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import DPTForDepthEstimation
except ImportError:  # optional dependency
    DPTForDepthEstimation = None

try:
    from vggt.models.vggt import VGGT
except ImportError:  # optional dependency
    VGGT = None


@dataclass
class SceneOutput:
    depth: torch.Tensor  # [B, T, H, W]
    confidence: Optional[torch.Tensor] = None  # [B, T, H, W]
    world_points: Optional[torch.Tensor] = None  # [B, T, H, W, 3]


class VGGTDepthModel(nn.Module):
    """Unified wrapper for official VGGT or a lighter DPT fallback."""

    def __init__(self, model_name: str, backend: str = "vggt") -> None:
        super().__init__()
        self.backend = backend
        self.model_name = model_name

        if backend == "vggt":
            if VGGT is None:
                raise ImportError(
                    "The official VGGT package is required. Install with: "
                    "pip install git+https://github.com/facebookresearch/vggt.git"
                )
            self.backbone = VGGT.from_pretrained(model_name)
        elif backend == "dpt":
            if DPTForDepthEstimation is None:
                raise ImportError(
                    "transformers is required for the DPT fallback. Install with: pip install transformers"
                )
            self.backbone = DPTForDepthEstimation.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model backend: {backend}")

    def forward(self, pixel_values: torch.Tensor) -> SceneOutput:
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() != 5:
            raise ValueError(f"Expected [B, T, 3, H, W] or [T, 3, H, W], got {tuple(pixel_values.shape)}")

        if self.backend == "vggt":
            predictions = self._forward_vggt(pixel_values)
            return SceneOutput(
                depth=predictions["depth"].squeeze(-1),
                confidence=predictions.get("depth_conf"),
                world_points=predictions.get("world_points"),
            )

        batch_size, sequence_length, channels, height, width = pixel_values.shape
        flat_frames = pixel_values.reshape(batch_size * sequence_length, channels, height, width)
        outputs = self.backbone(pixel_values=flat_frames)
        depth = outputs.predicted_depth.reshape(batch_size, sequence_length, *outputs.predicted_depth.shape[-2:])
        confidence = torch.sigmoid(-torch.abs(depth - depth.mean(dim=(-1, -2), keepdim=True)))
        return SceneOutput(depth=depth, confidence=confidence, world_points=None)

    def _forward_vggt(self, pixel_values: torch.Tensor) -> dict:
        if not pixel_values.is_cuda:
            return self.backbone(pixel_values)

        capability_major = torch.cuda.get_device_capability(pixel_values.device)[0]
        autocast_dtype = torch.bfloat16 if capability_major >= 8 else torch.float16
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            return self.backbone(pixel_values)
