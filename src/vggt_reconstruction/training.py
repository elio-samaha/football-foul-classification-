from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import FinetuneConfig
from .dataset import SoccerNetMVFoulDataset
from .model import VGGTDepthModel


class VGGTFinetuner:
    """Optional fine-tuning loop.

    Notes:
    - This expects pseudo/ground-truth depth maps stored as .npy files named like frame_XXXX_depth.npy
      alongside RGB frames.
    - If depth maps are missing, sample is skipped.
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model = VGGTDepthModel(
            model_name=config.init_model_name,
            backend=config.backend,
        ).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.train_ds = SoccerNetMVFoulDataset(
            split_file=config.train_manifest,
            sequence_length=config.sequence_length,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
        )
        self.val_ds = SoccerNetMVFoulDataset(
            split_file=config.val_manifest,
            sequence_length=config.sequence_length,
            frame_stride=config.frame_stride,
            resize_hw=config.resize_hw,
        )

    def fit(self) -> None:
        scaler = torch.cuda.amp.GradScaler(enabled=self.config.mixed_precision)
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
        dataset: SoccerNetMVFoulDataset,
        scaler: torch.cuda.amp.GradScaler | None,
        train: bool,
    ) -> float:
        total_loss = 0.0
        n = 0
        for sample in dataset:
            frames = sample.frames.to(self.device)
            for frame_idx, frame_path in enumerate(sample.frame_paths):
                depth_path = frame_path.with_name(frame_path.stem + "_depth.npy")
                if not depth_path.exists():
                    continue
                target = torch.from_numpy(__import__("numpy").load(depth_path)).float().to(self.device)
                if target.dim() == 2:
                    target = target.unsqueeze(0)
                pred_in = frames[frame_idx].unsqueeze(0)
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    pred = self.model(pred_in).depth
                    pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
                    loss = F.l1_loss(pred.squeeze(0), target)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    if self.config.max_grad_norm is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                total_loss += loss.item()
                n += 1
        return total_loss / max(1, n)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        out_dir = Path(self.config.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "val_loss": val_loss}, path)
