"""Reconstruction pipeline: MVFoul actions -> VGGT -> 3D point clouds."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ReconstructionConfig
from .dataset import MVFoulDataset, ActionSample
from .geometry import world_points_to_point_cloud, write_ply
from .model import VGGTModel


class ReconstructionPipeline:
    """Process MVFoul actions through the VGGT model and export 3D point clouds."""

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.dataset = MVFoulDataset(
            dataset_root=config.dataset_root,
            split=config.split,
            num_frames=config.num_frames_per_clip,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
            max_actions=config.max_actions,
        )

        print(f"Loading VGGT model: {config.vggt_model_name} ...")
        self.model = VGGTModel(
            model_name=config.vggt_model_name,
            device=config.device,
            dtype=config.dtype,
        ).to(self.device)
        self.model.eval()
        print("Model loaded.")

    @torch.no_grad()
    def run(self) -> dict:
        """Run reconstruction on all actions and return a summary dict."""
        results = {}
        n_actions = len(self.dataset)
        print(f"Processing {n_actions} actions from split '{self.config.split}' ...")

        for i in range(n_actions):
            action: ActionSample = self.dataset[i]
            action_dir = self.config.output_root / action.action_id
            action_dir.mkdir(parents=True, exist_ok=True)
            action_result = {"views": {}}

            for view_name, view_frames in action.views.items():
                view_dir = action_dir / view_name
                view_dir.mkdir(parents=True, exist_ok=True)

                images = view_frames.to(self.device)  # [T, 3, H, W]
                output = self.model(images)

                n_frames = images.shape[0]
                ply_paths = []
                for t in range(n_frames):
                    cloud = world_points_to_point_cloud(
                        world_points=output.world_points[0, t],
                        rgb_frame=images[t],
                        confidence=output.world_points_conf[0, t],
                        confidence_threshold=self.config.confidence_threshold,
                    )
                    ply_path = view_dir / f"frame_{t:03d}.ply"
                    write_ply(cloud, ply_path)
                    ply_paths.append(str(ply_path))

                if output.extrinsics is not None:
                    cam_data = {
                        "extrinsics": output.extrinsics[0].cpu().tolist(),
                        "intrinsics": output.intrinsics[0].cpu().tolist(),
                    }
                    cam_path = view_dir / "cameras.json"
                    with cam_path.open("w") as f:
                        json.dump(cam_data, f, indent=2)

                action_result["views"][view_name] = {
                    "n_frames": n_frames,
                    "ply_files": ply_paths,
                }

            if action.annotations:
                action_result["annotations"] = action.annotations
                ann_path = action_dir / "annotations.json"
                with ann_path.open("w") as f:
                    json.dump(action.annotations, f, indent=2)

            results[action.action_id] = action_result
            print(
                f"  [{i+1}/{n_actions}] {action.action_id}: "
                f"{len(action.views)} views processed"
            )

        summary_path = self.config.output_root / "summary.json"
        with summary_path.open("w") as f:
            json.dump(results, f, indent=2)

        print(f"Done. Results saved to {self.config.output_root}")
        return results
