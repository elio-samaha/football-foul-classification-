# football-foul-classification-

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

## Notes
- Start with pre-trained inference if depth supervision is not yet ready.
- Keep per-frame PLY outputs; they are useful for temporal 3D reasoning in foul classification.
