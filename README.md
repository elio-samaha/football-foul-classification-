# football-foul-classification-

Stage **4.2 / Stage 1** scaffold for **VGGT-style scene reconstruction** on **SoccerNet-MVFoul**.

This repo currently focuses on producing **per-frame 3D point clouds** from clip frames and providing an optional fine-tuning loop for depth prediction.

---

## 1) What is in the repo

- `src/vggt_reconstruction/`
  - `config.py`: inference/fine-tuning dataclasses.
  - `dataset.py`: temporal SoccerNet MVFoul clip loader from CSV manifest.
  - `model.py`: depth model wrapper (`Intel/dpt-hybrid-midas` by default).
  - `geometry.py`: depth-to-point-cloud projection + `.ply` export.
  - `pipeline.py`: end-to-end per-frame reconstruction pipeline.
  - `training.py`: optional fine-tuning loop using `*_depth.npy` sidecar targets.
- `scripts/run_reconstruction.py`: reconstruction entrypoint.
- `scripts/finetune_vggt.py`: fine-tuning entrypoint.
- `configs/*.csv`: example manifests.
- `docs/stage_4_2_stage_1_vggt_plan.md`: implementation plan.
- `docs/next_stage_handover.md`: colleague handover guide for the next stage.

---

## 2) Requirements

Python 3.10+ recommended.

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Install dev/test dependencies:

```bash
pip install -r requirements-dev.txt
```

---

## 3) Dataset setup (manual steps)

You need to prepare SoccerNet-MVFoul frames locally.

### Expected layout

```text
data/
  SoccerNet-MVFoul/
    frames/
      clip_0001/
        000000.jpg
        000001.jpg
        ...
      clip_0002/
        ...
    labels/
      clip_0001.json
      clip_0002.json
```

### Manifests

Create/edit these CSV files:

- `configs/reconstruction_split.csv`
- `configs/train_split.csv`
- `configs/val_split.csv`

Format (one row per clip window):

```text
clip_id,frames_dir,start_idx,num_frames,label_json_path
```

Example:

```text
clip_0001,data/SoccerNet-MVFoul/frames/clip_0001,0,64,data/SoccerNet-MVFoul/labels/clip_0001.json
```

---

## 4) Run reconstruction (pre-trained)

```bash
python scripts/run_reconstruction.py
```

Outputs:

```text
outputs/reconstruction/<clip_id>/frame_000.ply
outputs/reconstruction/<clip_id>/frame_001.ply
...
```

These are per-frame point clouds preserving temporal ordering.

---

## 5) Optional fine-tuning

Fine-tuning expects depth sidecar files for each RGB frame:

```text
frame_0001.jpg
frame_0001_depth.npy
```

Then run:

This repository now includes **Stage 4.2 / Stage 1** scaffolding for VGGT-style scene reconstruction on SoccerNet MVFoul data.

## What is implemented
- SoccerNet MVFoul clip loader (sequence-aware).
- Transformer depth wrapper (`Intel/dpt-hybrid-midas`) as a practical VGGT proxy.
- Per-frame depth-to-point-cloud projection.
- Optional fine-tuning loop using `*_depth.npy` files when available.

## Folder structure
- `src/vggt_reconstruction/`: core modules.
- `scripts/run_reconstruction.py`: inference entrypoint.
- `scripts/finetune_vggt.py`: fine-tuning entrypoint.
- `configs/*.csv`: example split manifests.
- `docs/stage_4_2_stage_1_vggt_plan.md`: implementation + fine-tuning plan.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio transformers opencv-python numpy
python scripts/run_reconstruction.py
```

## Fine-tuning
Store depth supervision next to frames as:
- `frame_0001.jpg`
- `frame_0001_depth.npy`

Then run:
```bash
python scripts/finetune_vggt.py
```

Checkpoints are written to:

```text
checkpoints/vggt_depth/
```

---

## 6) Verify the repo locally

```bash
python -m compileall src scripts tests
pytest -q
```

---

## 7) Known current limitations

- Camera intrinsics are fixed defaults in config and should be calibrated for better metric geometry.
- Current output is per-frame clouds; cross-frame registration/fusion is a next-stage task.
- Depth fine-tuning currently uses L1 depth loss only; temporal losses are planned next.

---

## 8) Next-stage roadmap

See `docs/next_stage_handover.md` for:
- exact takeover instructions,
- what to improve first,
- how to integrate with the foul classifier pipeline.
## Notes
- Start with pre-trained inference if depth supervision is not yet ready.
- Keep per-frame PLY outputs; they are useful for temporal 3D reasoning in foul classification.
