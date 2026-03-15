from __future__ import annotations

import torch

from .config import ReconstructionConfig
from .dataset import SoccerNetMVFoulDataset
from .geometry import depth_to_point_cloud, write_ply
from .model import VGGTDepthModel


class ReconstructionPipeline:
    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        self.dataset = SoccerNetMVFoulDataset(
            split_file=config.split_file,
            sequence_length=config.sequence_length,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
        )
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = VGGTDepthModel(model_name=config.model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def run(self) -> None:
        for clip in self.dataset:
            clip_output_dir = self.config.output_root / clip.clip_id
            clip_output_dir.mkdir(parents=True, exist_ok=True)

            frames = clip.frames.to(self.device)
            for t in range(frames.shape[0]):
                frame = frames[t].unsqueeze(0)
                out = self.model(frame)
                cloud = depth_to_point_cloud(
                    depth_map=out.depth[0],
                    rgb_frame=clip.frames[t],
                    fx=self.config.intrinsics_fx,
                    fy=self.config.intrinsics_fy,
                    cx=self.config.intrinsics_cx,
                    cy=self.config.intrinsics_cy,
                    confidence=out.confidence[0] if out.confidence is not None else None,
                    confidence_threshold=self.config.confidence_threshold,
                )
                write_ply(cloud, clip_output_dir / f"frame_{t:03d}.ply")
