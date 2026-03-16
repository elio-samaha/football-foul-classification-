from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import DPTForDepthEstimation
except ImportError:  # optional dependency
    DPTForDepthEstimation = None


@dataclass
class DepthOutput:
    depth: torch.Tensor  # [B, 1, H, W]
    confidence: Optional[torch.Tensor] = None


class VGGTDepthModel(nn.Module):
    """A practical VGGT-stage proxy using a pre-trained depth transformer.

    This wrapper keeps the interface stable so you can later swap to an
    official VGGT checkpoint without changing dataset/pipeline code.
    """

    def __init__(self, model_name: str = "Intel/dpt-hybrid-midas") -> None:
        super().__init__()
        if DPTForDepthEstimation is None:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        self.backbone = DPTForDepthEstimation.from_pretrained(model_name)

    def forward(self, pixel_values: torch.Tensor) -> DepthOutput:
        outputs = self.backbone(pixel_values=pixel_values)
        depth = outputs.predicted_depth.unsqueeze(1)
        confidence = torch.sigmoid(-torch.abs(depth - depth.mean(dim=(-1, -2), keepdim=True)))
        return DepthOutput(depth=depth, confidence=confidence)
