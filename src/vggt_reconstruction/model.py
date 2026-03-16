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
    # Official VGGT implementation from facebookresearch/vggt
    from vggt.models import VGGT  # type: ignore
except ImportError:  # optional dependency
    VGGT = None


@dataclass
class DepthOutput:
    depth: torch.Tensor  # [B, 1, H, W]
    confidence: Optional[torch.Tensor] = None


class VGGTDepthModel(nn.Module):
    """Depth head with pluggable backends (DPT proxy or real VGGT).

    The default backend is a DPT-based transformer published via Hugging Face.
    You can enable a real VGGT backbone by setting ``backend=\"vggt\"`` in
    the configuration (see :class:`ReconstructionConfig` / :class:`FinetuneConfig`).
    """

    def __init__(
        self,
        model_name: str = "Intel/dpt-hybrid-midas",
        backend: str = "dpt",
    ) -> None:
        super().__init__()
        backend = backend.lower()
        self.backend = backend
        self.model_name = model_name

        if backend == "dpt":
            if DPTForDepthEstimation is None:
                raise ImportError(
                    "transformers is required for the DPT backend. "
                    "Install with: pip install transformers"
                )
            self.backbone = DPTForDepthEstimation.from_pretrained(model_name)
        elif backend == "vggt":
            if VGGT is None:
                raise ImportError(
                    "The 'vggt' package is required for the real VGGT backend. "
                    "Install it with `pip install git+https://github.com/facebookresearch/vggt.git`."
                )
            # Use the official VGGT class. Model names follow the VGGT repo
            # (e.g. 'VGGT_Base', 'VGGT_Large', 'VGGT-1B-Commercial').
            self.backbone = VGGT.from_pretrained(model_name)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'dpt' or 'vggt'.")

    def forward(self, pixel_values: torch.Tensor) -> DepthOutput:
        if self.backend == "dpt":
            outputs = self.backbone(pixel_values=pixel_values)
            depth = outputs.predicted_depth.unsqueeze(1)
        else:  # self.backend == "vggt"
            # VGGT operates on batches of images [B, C, H, W].
            # The official implementation exposes multiple dense outputs; we
            # prioritise depth maps when available.
            predictions = self.backbone(images=pixel_values)
            if isinstance(predictions, dict) and "depth" in predictions:
                depth = predictions["depth"]
            elif isinstance(predictions, dict) and "point_map" in predictions:
                # As a fallback, approximate per-pixel depth from 3D point maps.
                point_map = predictions["point_map"]
                depth = point_map[..., 2:].permute(0, 3, 1, 2)  # [B, 1, H, W]
            else:
                raise RuntimeError(
                    "VGGT forward did not return a 'depth' or 'point_map' tensor. "
                    "Please check the installed VGGT version."
                )

        confidence = torch.sigmoid(-torch.abs(depth - depth.mean(dim=(-1, -2), keepdim=True)))
        return DepthOutput(depth=depth, confidence=confidence)

