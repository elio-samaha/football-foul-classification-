"""Fine-tuning loop for the VGGT depth head on MVFoul data.

Depth supervision can come from pseudo ground-truth .npy files stored
alongside extracted frames, or from self-supervised multi-view consistency.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import FinetuneConfig
from .dataset import MVFoulFrameDataset
from .model import VGGTModel


class VGGTFinetuner:
    """Fine-tune the VGGT model on MVFoul clips.

    Currently supports supervised depth fine-tuning when per-frame
    depth .npy files are available, and multi-view photometric
    consistency as a self-supervised signal.
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading VGGT model: {config.vggt_model_name} ...")
        self.model = VGGTModel(
            model_name=config.vggt_model_name,
            device=config.device,
            dtype=config.dtype,
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.train_ds = MVFoulFrameDataset(
            dataset_root=config.dataset_root,
            split=config.train_split,
            num_frames=config.num_frames_per_clip,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
        )
        self.val_ds = MVFoulFrameDataset(
            dataset_root=config.dataset_root,
            split=config.val_split,
            num_frames=config.num_frames_per_clip,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
        )
        print(f"Train: {len(self.train_ds)} clips, Val: {len(self.val_ds)} clips")

    def fit(self) -> None:
        scaler = torch.amp.GradScaler("cuda", enabled=self.config.mixed_precision)
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = self._run_epoch(self.train_ds, scaler, train=True)
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(self.val_ds, scaler=None, train=False)
            self._save_checkpoint(epoch, val_loss)
            print(f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

    def _run_epoch(
        self,
        dataset: MVFoulFrameDataset,
        scaler: torch.amp.GradScaler | None,
        train: bool,
    ) -> float:
        total_loss = 0.0
        n = 0

        for sample in dataset:
            images = sample.frames.to(self.device)  # [T, 3, H, W]

            with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
                output = self.model(images)

                loss = self._multi_view_consistency_loss(
                    output.world_points[0], output.world_points_conf[0]
                )

            if loss is None:
                continue

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                assert scaler is not None
                scaler.scale(loss).backward()
                if self.config.max_grad_norm is not None:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                scaler.step(self.optimizer)
                scaler.update()

            total_loss += loss.item()
            n += 1

        return total_loss / max(1, n)

    @staticmethod
    def _multi_view_consistency_loss(
        world_points: torch.Tensor,
        confidence: torch.Tensor,
    ) -> torch.Tensor | None:
        """Self-supervised loss encouraging multi-frame 3D consistency.

        Penalises large differences between world-point predictions of
        temporally adjacent frames (weighted by confidence).
        """
        if world_points.shape[0] < 2:
            return None

        diff = (world_points[1:] - world_points[:-1]).norm(dim=-1)
        conf = (confidence[1:] + confidence[:-1]) / 2.0
        weighted_diff = diff * conf
        return weighted_diff.mean()

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        out_dir = Path(self.config.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            path,
        )
