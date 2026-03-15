import numpy as np
import torch

from src.vggt_reconstruction.geometry import depth_to_point_cloud


def test_depth_to_point_cloud_shapes() -> None:
    depth = torch.ones(1, 4, 4)
    rgb = torch.zeros(3, 4, 4)
    cloud = depth_to_point_cloud(depth, rgb, fx=1.0, fy=1.0, cx=0.0, cy=0.0)
    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[1] == 6
