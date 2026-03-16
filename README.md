# Football Foul Classification — VGGT 3D Reconstruction

This repository uses the **VGGT** (Visual Geometry Grounded Transformer, CVPR 2025 Best Paper) model from Meta AI Research to perform 3D scene reconstruction on the **SoccerNet MVFoul** multi-view foul dataset.

## What is implemented

- **MVFoul dataset integration**: automatic discovery and loading of multi-view video clips from the SoccerNet MVFoul dataset structure (`Train/`, `Valid/`, `Test/`, `Chall/` splits).
- **Real VGGT model** (`facebook/VGGT-1B`): produces depth maps, 3D world point maps, camera extrinsics/intrinsics, and confidence scores from multiple views.
- **Per-frame 3D point cloud export**: each frame is projected to a colored XYZRGB point cloud (PLY format) using VGGT's world-point predictions.
- **Camera parameter export**: estimated extrinsics and intrinsics saved per view as JSON.
- **Fine-tuning loop**: multi-view consistency self-supervised loss for adapting VGGT to soccer footage.
- **Dataset download script**: supports both the SoccerNet API and HuggingFace Hub.

## Folder structure

```
src/vggt_reconstruction/     Core modules (dataset, model, geometry, pipeline, training)
scripts/download_mvfoul.py   Download the MVFoul dataset
scripts/run_reconstruction.py   Reconstruction inference entrypoint
scripts/finetune_vggt.py     Fine-tuning entrypoint
tests/                       Unit tests
configs/                     Legacy split manifests (optional)
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the MVFoul dataset

Via HuggingFace Hub (recommended):

```bash
python scripts/download_mvfoul.py --dest data/mvfoul --unzip
```

Via SoccerNet API (requires NDA password from https://www.soccer-net.org/data):

```bash
python scripts/download_mvfoul.py --dest data/mvfoul --backend soccernet --password <YOUR_NDA_PASSWORD> --unzip
```

### 3. Run reconstruction

```bash
python scripts/run_reconstruction.py \
    --dataset-root data/mvfoul \
    --split train \
    --max-actions 5 \
    --device cuda
```

This processes each action's multi-view clips through VGGT and exports per-frame PLY point clouds and camera parameters to `outputs/reconstruction/`.

### 4. Fine-tuning (optional)

```bash
python scripts/finetune_vggt.py \
    --dataset-root data/mvfoul \
    --epochs 5 \
    --device cuda
```

## Model

This project uses **VGGT-1B** ([github.com/facebookresearch/vggt](https://github.com/facebookresearch/vggt), [Hugging Face](https://huggingface.co/facebook/VGGT-1B)), which:

- Takes one or multiple image views as input
- Predicts depth maps, dense 3D world-point maps, camera extrinsics, and intrinsics
- Completes reconstruction in under one second per clip
- Won the CVPR 2025 Best Paper Award

## Dataset

The [SoccerNet MVFoul](https://github.com/SoccerNet/sn-mvfoul) dataset contains 3,901 multi-view foul actions:

| Split      | Actions |
|------------|---------|
| Train      | 2,916   |
| Valid      | 411     |
| Test       | 301     |
| Challenge  | 273     |

Each action has 2+ camera views (mp4 clips ~5 seconds each) annotated with 10 foul properties by a professional referee.

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
