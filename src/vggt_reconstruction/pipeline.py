from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import ReconstructionConfig
from .dataset import SoccerNetMVFoulDataset, MVFoulVideoDataset, MultiViewSample, ClipSample
from .geometry import depth_to_point_cloud, world_points_to_point_cloud, write_ply
from .model import VGGTModel, DPTDepthModel, load_model


class ReconstructionPipeline:
    """Run 3D reconstruction on MVFoul clips using VGGT or DPT fallback.

    Supports two dataset modes:
    - CSV-based: pre-extracted frames referenced by a split CSV.
    - Video-based: raw .mp4 files in the standard MVFoul directory layout.
    """

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = load_model(
            model_name=config.model_name,
            use_vggt=config.use_vggt,
            dtype=config.dtype,
            device=str(self.device),
        )
        self.use_vggt = config.use_vggt

    def run_on_csv(self, split_file: Path | None = None) -> None:
        """Reconstruct clips listed in a CSV manifest."""
        sf = split_file or self.config.split_file
        dataset = SoccerNetMVFoulDataset(
            split_file=sf,
            sequence_length=self.config.sequence_length,
            frame_stride=self.config.frame_stride,
            resize_hw=self.config.resize_hw,
        )
        print(f"Processing {len(dataset)} clips from {sf}")
        for i in range(len(dataset)):
            clip = dataset[i]
            self._reconstruct_clip(clip)

    def run_on_videos(self, split: str = "train") -> None:
        """Reconstruct directly from raw MVFoul videos."""
        dataset = MVFoulVideoDataset(
            dataset_root=self.config.dataset_root,
            split=split,
            sequence_length=self.config.sequence_length,
            frame_stride=self.config.frame_stride,
            resize_hw=self.config.resize_hw,
        )
        print(f"Processing {len(dataset)} actions from {split} split")
        for i in range(len(dataset)):
            mv_sample = dataset[i]
            self._reconstruct_multiview(mv_sample)

    def run(self) -> None:
        """Default entry point: use CSV if available, else discover videos."""
        if self.config.split_file and self.config.split_file.exists():
            self.run_on_csv()
        else:
            for split in ("train", "valid", "test"):
                self.run_on_videos(split)

    @torch.no_grad()
    def _reconstruct_clip(self, clip: ClipSample) -> None:
        """Reconstruct a single clip (one camera view)."""
        clip_dir = self.config.output_root / clip.clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        frames = clip.frames.to(self.device)

        if self.use_vggt and isinstance(self.model, VGGTModel):
            self._run_vggt_on_frames(frames, clip.frames, clip_dir)
        else:
            self._run_dpt_on_frames(frames, clip.frames, clip_dir)

        print(f"  {clip.clip_id}: {frames.shape[0]} frames reconstructed")

    @torch.no_grad()
    def _reconstruct_multiview(self, mv_sample: MultiViewSample) -> None:
        """Reconstruct a multi-view action (all camera views fed together to VGGT)."""
        action_dir = self.config.output_root / mv_sample.action_id
        action_dir.mkdir(parents=True, exist_ok=True)

        if not mv_sample.clips:
            return

        if self.use_vggt and isinstance(self.model, VGGTModel):
            all_frames = mv_sample.all_frames.to(self.device)
            out = self.model(all_frames)

            frame_offset = 0
            for clip in mv_sample.clips:
                clip_dir = action_dir / clip.clip_id.split("_")[-1]
                clip_dir.mkdir(parents=True, exist_ok=True)
                n = clip.frames.shape[0]

                for t in range(n):
                    gi = frame_offset + t
                    wp = out.world_points[0, gi]   # [H, W, 3]
                    wc = out.world_points_conf[0, gi] if out.world_points_conf is not None else None
                    rgb = clip.frames[t]

                    cloud = world_points_to_point_cloud(
                        wp, rgb,
                        confidence=wc,
                        confidence_threshold=self.config.confidence_threshold,
                    )
                    write_ply(cloud, clip_dir / f"frame_{t:03d}.ply")

                frame_offset += n

            self._save_camera_info(out, action_dir)
        else:
            for clip in mv_sample.clips:
                self._reconstruct_clip(clip)

        print(f"  {mv_sample.action_id}: {len(mv_sample.clips)} views reconstructed")

    def _run_vggt_on_frames(
        self,
        frames_device: torch.Tensor,
        frames_cpu: torch.Tensor,
        output_dir: Path,
    ) -> None:
        """Use VGGT on a sequence of frames from a single clip."""
        out = self.model(frames_device)

        for t in range(frames_device.shape[0]):
            if out.world_points is not None:
                cloud = world_points_to_point_cloud(
                    out.world_points[0, t],
                    frames_cpu[t],
                    confidence=out.world_points_conf[0, t] if out.world_points_conf is not None else None,
                    confidence_threshold=self.config.confidence_threshold,
                )
            else:
                depth = out.depth[0, t]
                conf = out.depth_conf[0, t] if out.depth_conf is not None else None
                cloud = depth_to_point_cloud(
                    depth, frames_cpu[t],
                    fx=self.config.intrinsics_fx,
                    fy=self.config.intrinsics_fy,
                    cx=self.config.intrinsics_cx,
                    cy=self.config.intrinsics_cy,
                    confidence=conf,
                    confidence_threshold=self.config.confidence_threshold,
                )
            write_ply(cloud, output_dir / f"frame_{t:03d}.ply")

        self._save_camera_info(out, output_dir)

    def _run_dpt_on_frames(
        self,
        frames_device: torch.Tensor,
        frames_cpu: torch.Tensor,
        output_dir: Path,
    ) -> None:
        """Use DPT depth model as fallback (per-frame)."""
        for t in range(frames_device.shape[0]):
            frame = frames_device[t].unsqueeze(0)
            out = self.model(frame)
            cloud = depth_to_point_cloud(
                depth_map=out.depth[0],
                rgb_frame=frames_cpu[t],
                fx=self.config.intrinsics_fx,
                fy=self.config.intrinsics_fy,
                cx=self.config.intrinsics_cx,
                cy=self.config.intrinsics_cy,
                confidence=out.depth_conf[0] if out.depth_conf is not None else None,
                confidence_threshold=self.config.confidence_threshold,
            )
            write_ply(cloud, output_dir / f"frame_{t:03d}.ply")

    def _save_camera_info(self, vggt_out, output_dir: Path) -> None:
        """Save VGGT-predicted camera poses alongside point clouds."""
        if vggt_out.camera_poses is not None:
            poses = vggt_out.camera_poses[0].detach().cpu().numpy()
            np.save(output_dir / "camera_poses.npy", poses)

        meta = {
            "model": self.config.model_name,
            "use_vggt": self.use_vggt,
            "sequence_length": self.config.sequence_length,
            "frame_stride": self.config.frame_stride,
            "resize_hw": list(self.config.resize_hw),
            "confidence_threshold": self.config.confidence_threshold,
        }
        with open(output_dir / "reconstruction_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
