from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import FinetuneConfig
from .dataset import SoccerNetMVFoulDataset
from .model import VGGTModel, DPTDepthModel, load_model


class VGGTFinetuner:
    """Fine-tuning loop for VGGT or DPT depth backbone.

    Expects pseudo/ground-truth depth stored as *_depth.npy alongside RGB frames.
    Samples without depth supervision are skipped.
    """

    def __init__(self, config: FinetuneConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        self.model = load_model(
            model_name=config.init_model_name,
            use_vggt=config.use_vggt,
            dtype=config.dtype,
            device=str(self.device),
        )
        self.model.train()

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
        dataset: SoccerNetMVFoulDataset,
        scaler: torch.amp.GradScaler | None,
        train: bool,
    ) -> float:
        total_loss = 0.0
        n = 0
        for i in range(len(dataset)):
            sample = dataset[i]
            frames = sample.frames.to(self.device)
            for frame_idx, frame_path in enumerate(sample.frame_paths):
                depth_path = frame_path.with_name(frame_path.stem + "_depth.npy")
                if not depth_path.exists():
                    continue
                target = torch.from_numpy(np.load(depth_path)).float().to(self.device)
                if target.dim() == 2:
                    target = target.unsqueeze(0)

                if isinstance(self.model, VGGTModel):
                    pred_in = frames[frame_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]
                else:
                    pred_in = frames[frame_idx].unsqueeze(0)  # [1, 3, H, W]

                with torch.amp.autocast("cuda", enabled=self.config.mixed_precision):
                    out = self.model(pred_in)
                    pred_depth = out.depth
                    if pred_depth.dim() == 5:
                        pred_depth = pred_depth[0, 0, :, :, 0].unsqueeze(0)
                    elif pred_depth.dim() == 4:
                        pred_depth = pred_depth.squeeze(0)
                    pred_depth = F.interpolate(
                        pred_depth.unsqueeze(0) if pred_depth.dim() == 2 else pred_depth.unsqueeze(0),
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    loss = F.l1_loss(pred_depth.squeeze(0), target)

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

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        out_dir = Path(self.config.checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"epoch_{epoch:03d}_valloss_{val_loss:.4f}.pt"
        torch.save(
            {"model": self.model.state_dict(), "epoch": epoch, "val_loss": val_loss},
            path,
        )
