# Football Foul Classification with VGGT 3D Reconstruction

This repository implements **VGGT-based 3D scene reconstruction** on the [SoccerNet MVFoul](https://github.com/SoccerNet/sn-mvfoul) multi-view foul dataset, producing per-frame point clouds and camera parameters for downstream foul classification.

## What is implemented

- **Real VGGT model** ([facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B)) for multi-view depth estimation, 3D point cloud reconstruction, and camera parameter prediction.
- **SoccerNet MVFoul integration**: download script, frame extraction, and data loading for the MVFoul multi-view foul dataset (3,901 actions with multiple camera angles).
- **Two dataset modes**: CSV-based (pre-extracted frames) and direct video loading from raw `.mp4` files.
- **DPT fallback** (`Intel/dpt-hybrid-midas`) for environments without GPU or for quick testing.
- Per-frame depth-to-point-cloud projection and VGGT native world-point clouds.
- Optional fine-tuning loop using `*_depth.npy` supervision files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download the MVFoul Dataset

You need to sign the [SoccerNet NDA](https://docs.google.com/forms/d/e/1FAIpQLSfYFqjZNm4IgwGnyJXDPk2Ko_lZcbVtYX73w5lf6din5nxfmA/viewform) to get a download password, then:

```bash
python scripts/download_mvfoul.py --password YOUR_PASSWORD --output_dir data/mvfoul
```

After downloading, unzip each split folder (Train, Valid, Test, Chall) in place.

## Prepare Frames and Splits

Extract frames from videos and generate CSV manifests:

```bash
python scripts/prepare_mvfoul.py --dataset_dir data/mvfoul --fps 5
```

This creates:
- `data/mvfoul/frames/{split}/action_XXXX/clip_N/frame_NNNN.jpg`
- `configs/mvfoul_train.csv`, `configs/mvfoul_valid.csv`, `configs/mvfoul_test.csv`

## Run Reconstruction

### Using VGGT (recommended)
```bash
# From pre-extracted frames (CSV):
python scripts/run_reconstruction.py --split_file configs/mvfoul_train.csv

# Directly from raw videos:
python scripts/run_reconstruction.py --dataset_root data/mvfoul --split train
```

### Using DPT fallback
```bash
python scripts/run_reconstruction.py --no-vggt --split_file configs/mvfoul_train.csv
```

## Fine-tuning

If depth supervision is available (`frame_XXXX_depth.npy` next to frames):

```bash
python scripts/finetune_vggt.py --train_manifest configs/mvfoul_train.csv --val_manifest configs/mvfoul_valid.csv
```

## Project Structure

```
src/vggt_reconstruction/
    __init__.py          # Public API exports
    config.py            # ReconstructionConfig, FinetuneConfig, MVFoulAction
    dataset.py           # SoccerNetMVFoulDataset (CSV), MVFoulVideoDataset (raw videos)
    model.py             # VGGTModel (real VGGT-1B), DPTDepthModel (fallback)
    geometry.py          # Point cloud projection (VGGT world_points + depth fallback)
    pipeline.py          # ReconstructionPipeline (multi-view + single-clip)
    training.py          # VGGTFinetuner

scripts/
    download_mvfoul.py   # Download MVFoul via SoccerNet API
    prepare_mvfoul.py    # Extract frames, generate CSV splits
    run_reconstruction.py # Inference entrypoint
    finetune_vggt.py     # Fine-tuning entrypoint

configs/                 # CSV split manifests
tests/                   # Unit tests
```

## About the MVFoul Dataset

The SoccerNet-MVFoul dataset contains **3,901 multi-view foul actions**:
- Training: 2,916 actions
- Validation: 411 actions
- Test: 301 actions
- Challenge: 273 actions (no annotations)

Each action has multiple camera views annotated with **10 properties** including:
- **Offence severity**: No offence, Offence + No card, Offence + Yellow card, Offence + Red card
- **Action class**: Standing tackle, Tackle, Holding, Pushing, Challenge, Dive, High leg, Elbowing

## About VGGT

[VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry Grounded Transformer) won Best Paper at CVPR 2025. It predicts:
- Depth maps
- 3D world-point coordinates per pixel
- Camera extrinsics and intrinsics
- Point tracking

This makes it ideal for multi-view 3D reconstruction of foul scenes from multiple camera angles.

## Running Tests

```bash
python -m pytest tests/ -v
```
