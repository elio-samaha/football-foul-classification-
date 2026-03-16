"""Tests for the VGGT model wrapper.

These tests verify the module structure and imports without requiring
a GPU or model download.
"""

import pytest

from src.vggt_reconstruction.model import VGGTModel, VGGTOutput, _VGGT_AVAILABLE


def test_vggt_available() -> None:
    """VGGT package should be importable."""
    assert _VGGT_AVAILABLE is True


def test_vggt_output_dataclass() -> None:
    """VGGTOutput should be a proper dataclass."""
    import torch

    out = VGGTOutput(
        depth=torch.zeros(1, 2, 4, 4, 1),
        depth_conf=torch.zeros(1, 2, 4, 4),
        world_points=torch.zeros(1, 2, 4, 4, 3),
        world_points_conf=torch.zeros(1, 2, 4, 4),
        pose_enc=torch.zeros(1, 2, 9),
    )
    assert out.depth.shape == (1, 2, 4, 4, 1)
    assert out.extrinsics is None
    assert out.intrinsics is None
    assert isinstance(out.raw, dict)
