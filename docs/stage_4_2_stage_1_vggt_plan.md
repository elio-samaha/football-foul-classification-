# Stage 4.2 / Stage 1 — VGGT-based Scene Reconstruction Plan

## Goal
Build a robust 3D reconstruction module for SoccerNet MVFoul clips using the
real VGGT model (facebook/VGGT-1B) that:
1. Loads multi-view video clips directly from the MVFoul dataset structure.
2. Predicts depth, 3D world points, and camera parameters per frame using VGGT.
3. Exports per-frame colored 3D point clouds (PLY) and camera parameters (JSON).
4. Supports fine-tuning with multi-view consistency loss.

## Implementation

### Model
- **VGGT-1B** (facebook/VGGT-1B) — CVPR 2025 Best Paper.
- Predicts: depth maps, 3D world point maps, camera extrinsics/intrinsics, confidence.
- Wrapped in `VGGTModel` class with structured `VGGTOutput` dataclass.

### Dataset
- **MVFoulDataset**: loads multi-view actions; each item yields all camera views.
- **MVFoulFrameDataset**: flattened version where each item is a single clip.
- Directly reads mp4 videos from the standard MVFoul directory layout.
- Supports all splits: Train, Valid, Test, Chall.

### Pipeline
- `ReconstructionPipeline`: iterates over actions, runs VGGT on each view,
  exports PLY point clouds and camera JSON per frame.

### Fine-tuning
- Self-supervised multi-view temporal consistency loss.
- Adjacent frames should predict similar world points (weighted by confidence).

## Data interface to SoccerNet MVFoul

```
<dataset_root>/
    Train/
        action_1/
            clip_1.mp4
            clip_2.mp4
        action_2/
            ...
    Valid/
        ...
    Test/
        ...
    Chall/
        ...
```

Download via:
```bash
python scripts/download_mvfoul.py --dest data/mvfoul --unzip
```
