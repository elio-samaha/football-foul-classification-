import torch
import pytest

from src.vggt_reconstruction.model import VGGTOutput


def test_vggt_output_fields() -> None:
    depth = torch.randn(1, 4, 32, 32, 1)
    out = VGGTOutput(depth=depth)
    assert out.depth.shape == (1, 4, 32, 32, 1)
    assert out.world_points is None
    assert out.camera_poses is None


def test_vggt_output_with_world_points() -> None:
    depth = torch.randn(1, 4, 32, 32, 1)
    wp = torch.randn(1, 4, 32, 32, 3)
    wp_conf = torch.rand(1, 4, 32, 32)
    poses = torch.randn(1, 4, 9)
    out = VGGTOutput(
        depth=depth,
        world_points=wp,
        world_points_conf=wp_conf,
        camera_poses=poses,
    )
    assert out.world_points.shape == (1, 4, 32, 32, 3)
    assert out.camera_poses.shape == (1, 4, 9)
