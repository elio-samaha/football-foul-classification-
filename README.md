# football-foul-classification-

This repository includes a VGGT-style reconstruction pipeline for SoccerNet MVFoul clips.

## What is implemented
- Downloader wrapper for the official SoccerNet MVFoul task.
- Manifest builder that scans downloaded MVFoul videos or extracted frame folders.
- Real VGGT backend support via `facebook/VGGT-1B`.
- DPT fallback backend via `Intel/dpt-hybrid-midas`.
- Per-frame and merged point-cloud export.
- Optional fine-tuning loop using `*_depth.npy` files when available.

## Folder structure
- `src/vggt_reconstruction/`: core modules.
- `scripts/download_mvfoul.py`: dataset download + manifest generation.
- `scripts/run_reconstruction.py`: inference entrypoint.
- `scripts/finetune_vggt.py`: fine-tuning entrypoint.
- `configs/*.csv`: generated or user-maintained manifests.
- `docs/stage_4_2_stage_1_vggt_plan.md`: implementation + fine-tuning plan.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision transformers SoccerNet opencv-python numpy
pip install git+https://github.com/facebookresearch/vggt.git
```

## Download MVFoul
The official MVFoul dataset requires an NDA/password from SoccerNet.

Once you have it:
```bash
export SOCCERNET_MVFOUL_PASSWORD="your-password"
python3 scripts/download_mvfoul.py \
  --soccernet-local-dir data/SoccerNet \
  --splits train valid test \
  --version 720p
```

This downloads into `data/SoccerNet/mvfouls/`, extracts the archives, and writes
`configs/reconstruction_split.csv`.

## Run reconstruction with the real VGGT model
```bash
python3 scripts/run_reconstruction.py \
  --dataset-root data/SoccerNet/mvfouls \
  --split-file configs/reconstruction_split.csv \
  --model-backend vggt \
  --model-name facebook/VGGT-1B
```

The pipeline will:
- sample a temporal window around the foul frame,
- run official VGGT inference on the clip,
- save `frame_XXX.ply`,
- save `frame_XXX_depth.npy`,
- save a merged `clip_merged.ply`.

## One-command download + inference
```bash
python3 scripts/run_reconstruction.py \
  --download-mvfoul \
  --mvfoul-password "$SOCCERNET_MVFOUL_PASSWORD" \
  --download-version 720p \
  --model-backend vggt
```

## Fine-tuning
Store depth supervision next to frames as:
- `frame_0001.jpg`
- `frame_0001_depth.npy`

Then run:
```bash
python3 scripts/finetune_vggt.py \
  --train-manifest configs/train_split.csv \
  --val-manifest configs/val_split.csv \
  --model-backend dpt
```

## Notes
- The real VGGT checkpoint is large; use `--model-backend dpt` if you need a lighter fallback.
- MVFoul manifests can point to either videos or pre-extracted frame folders.
- Keep per-frame PLY outputs; they are useful for temporal 3D reasoning in foul classification.
