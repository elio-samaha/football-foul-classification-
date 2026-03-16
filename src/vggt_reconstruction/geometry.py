from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def depth_to_point_cloud(
    depth_map: torch.Tensor,
    rgb_frame: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    confidence: torch.Tensor | None = None,
    confidence_threshold: float = 0.05,
) -> np.ndarray:
    """Project one depth frame into XYZRGB points (N, 6) using pinhole intrinsics.

    Used as fallback when VGGT world_points are not available.
    """
    depth = depth_map.squeeze().detach().cpu().numpy()
    rgb = rgb_frame.detach().cpu().permute(1, 2, 0).numpy()

    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]

    z = np.maximum(depth, 1e-6)
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    if confidence is not None:
        conf = confidence.squeeze().detach().cpu().numpy().reshape(-1)
        mask = conf > confidence_threshold
        points = points[mask]
        colors = colors[mask]

    return np.concatenate([points, colors], axis=1)


def world_points_to_point_cloud(
    world_points: torch.Tensor,
    rgb_frame: torch.Tensor,
    confidence: torch.Tensor | None = None,
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Convert VGGT world_points directly to an XYZRGB point cloud (N, 6).

    Args:
        world_points: [H, W, 3] world coordinates from VGGT.
        rgb_frame: [3, H, W] or [H, W, 3] RGB values in [0, 1].
        confidence: [H, W] confidence scores from VGGT.
        confidence_threshold: discard points below this confidence.
    """
    pts = world_points.detach().cpu().numpy().reshape(-1, 3)

    if rgb_frame.dim() == 3 and rgb_frame.shape[0] == 3:
        rgb = rgb_frame.detach().cpu().permute(1, 2, 0).numpy()
    else:
        rgb = rgb_frame.detach().cpu().numpy()
    colors = rgb.reshape(-1, 3)

    if confidence is not None:
        conf = confidence.detach().cpu().numpy().reshape(-1)
        mask = conf > confidence_threshold
        pts = pts[mask]
        colors = colors[mask]

    return np.concatenate([pts, colors], axis=1)


def write_ply(points_xyzrgb: np.ndarray, output_path: Path) -> None:
    """Write an XYZRGB point cloud to a PLY file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(points_xyzrgb)
    with output_path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for x, y, z, r, g, b in points_xyzrgb:
            f.write(
                f"{x:.6f} {y:.6f} {z:.6f} "
                f"{int(np.clip(r * 255, 0, 255))} "
                f"{int(np.clip(g * 255, 0, 255))} "
                f"{int(np.clip(b * 255, 0, 255))}\n"
            )
