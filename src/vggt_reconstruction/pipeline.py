from __future__ import annotations

import numpy as np
import torch

from .config import ReconstructionConfig
from .dataset import SoccerNetMVFoulDataset, build_mvfoul_manifest
from .geometry import depth_to_point_cloud, world_points_to_point_cloud, write_ply
from .model import VGGTDepthModel


class ReconstructionPipeline:
    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        split_file = self._resolve_split_file()
        self.dataset = SoccerNetMVFoulDataset(
            split_file=split_file,
            sequence_length=config.sequence_length,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
            resize_mode=config.resize_mode,
        )
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = VGGTDepthModel(model_name=config.model_name, backend=config.model_backend).to(self.device)
        self.model.eval()

    def _resolve_split_file(self):
        if self.config.split_file is not None and self._manifest_has_entries(self.config.split_file):
            return self.config.split_file
        if not self.config.auto_build_manifest:
            raise FileNotFoundError(f"Manifest not found: {self.config.split_file}")

        manifest_path = self.config.split_file or (self.config.output_root / "reconstruction_manifest.csv")
        return build_mvfoul_manifest(
            dataset_root=self.config.dataset_root,
            output_path=manifest_path,
            sequence_length=self.config.sequence_length,
            frame_stride=self.config.frame_stride,
            preferred_foul_frame=self.config.preferred_foul_frame,
            max_clips=self.config.max_clips,
        )

    @staticmethod
    def _manifest_has_entries(path) -> bool:
        if path is None or not path.exists():
            return False
        content_lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
        if not content_lines:
            return False
        if len(content_lines) == 1 and "clip_id" in content_lines[0] and "media_path" in content_lines[0]:
            return False
        return True

    @torch.no_grad()
    def run(self) -> None:
        for clip in self.dataset:
            clip_output_dir = self.config.output_root / clip.clip_id
            clip_output_dir.mkdir(parents=True, exist_ok=True)

            frames = clip.frames.to(self.device)
            out = self.model(frames)
            merged_clouds = []
            for t in range(frames.shape[0]):
                if out.world_points is not None:
                    cloud = world_points_to_point_cloud(
                        world_points=out.world_points[0, t],
                        rgb_frame=clip.frames[t],
                        confidence=out.confidence[0, t] if out.confidence is not None else None,
                        confidence_threshold=self.config.confidence_threshold,
                    )
                else:
                    cloud = depth_to_point_cloud(
                        depth_map=out.depth[0, t],
                        rgb_frame=clip.frames[t],
                        fx=self.config.intrinsics_fx,
                        fy=self.config.intrinsics_fy,
                        cx=self.config.intrinsics_cx,
                        cy=self.config.intrinsics_cy,
                        confidence=out.confidence[0, t] if out.confidence is not None else None,
                        confidence_threshold=self.config.confidence_threshold,
                    )
                write_ply(cloud, clip_output_dir / f"frame_{t:03d}.ply")
                if self.config.save_depth_maps:
                    np.save(clip_output_dir / f"frame_{t:03d}_depth.npy", out.depth[0, t].detach().cpu().numpy())
                merged_clouds.append(cloud)

            if merged_clouds:
                write_ply(np.concatenate(merged_clouds, axis=0), clip_output_dir / "clip_merged.ply")
