import numpy as np
import torch

from src.vggt_reconstruction.geometry import (
    depth_to_point_cloud,
    world_points_to_point_cloud,
)


def test_depth_to_point_cloud_shapes() -> None:
    depth = torch.ones(1, 4, 4)
    rgb = torch.zeros(3, 4, 4)
    cloud = depth_to_point_cloud(depth, rgb, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[1] == 6


def test_depth_to_point_cloud_with_confidence() -> None:
    depth = torch.ones(1, 4, 4)
    rgb = torch.zeros(3, 4, 4)
    conf = torch.ones(1, 4, 4) * 0.1
    cloud = depth_to_point_cloud(
        depth, rgb, fx=1.0, fy=1.0, cx=0.0, cy=0.0,
        confidence=conf, confidence_threshold=0.05,
    )
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[1] == 6
    assert cloud.shape[0] == 16


def test_world_points_to_point_cloud() -> None:
    world_pts = torch.randn(4, 4, 3)
    rgb = torch.rand(3, 4, 4)
    cloud = world_points_to_point_cloud(world_pts, rgb)
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape == (16, 6)


def test_world_points_with_confidence_filtering() -> None:
    world_pts = torch.randn(4, 4, 3)
    rgb = torch.rand(3, 4, 4)
    conf = torch.zeros(4, 4)
    conf[0, 0] = 1.0
    conf[1, 1] = 0.8
    cloud = world_points_to_point_cloud(
        world_pts, rgb, confidence=conf, confidence_threshold=0.5,
    )
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[0] == 2
    assert cloud.shape[1] == 6
