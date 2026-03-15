# Stage 4.2 / Stage 1 — VGGT-based Scene Reconstruction Plan

## Goal
Build a robust 3D reconstruction module for SoccerNet MVFoul clips that:
1. Takes a temporal sequence of frames as input.
2. Predicts depth per frame with a transformer-based geometry model.
3. Converts each frame into a colored 3D point cloud.
4. Exports both **per-frame point clouds** and a clip-level merged cloud for downstream foul classification.

## Recommended first implementation
- Use a pre-trained depth transformer (`Intel/dpt-hybrid-midas`) as a VGGT-stage proxy.
- Keep model interfaces abstracted so you can later plug in a true VGGT checkpoint.
- Start with per-frame reconstruction output (`frame_XXX.ply`) to preserve temporal structure.

## Data interface to SoccerNet MVFoul
The split CSV is the bridge between SoccerNet folders and training/inference:

```text
clip_id,frames_dir,start_idx,num_frames,label_json_path
```

This avoids hardcoding the dataset internals and lets you align reconstruction windows with your foul labels.

## Fine-tuning strategy

### Option A (now): pre-trained only
- Run zero-shot depth inference to produce pseudo-3D.
- Validate qualitatively (point cloud consistency over time).

### Option B (recommended next): supervised/pseudo-supervised fine-tuning
- If depth GT or pseudo-depth is available (`*_depth.npy`), train with L1 depth loss.
- Add temporal smoothness later:
  - photometric reprojection loss
  - adjacent-frame depth consistency
- Save checkpoints each epoch.

## Temporal modeling ideas for your class project
1. **Per-frame clouds + temporal links**: preserve each frame cloud and attach metadata (`timestamp`, `camera_id`).
2. **Sliding-window fused cloud**: register `T` neighboring clouds into one local scene.
3. **Hybrid output for classifier**:
   - 2D stream: RGB clip
   - 3D stream: sequence of point-cloud embeddings
   - Optional fusion transformer on top.

## Deliverables from this code
- `scripts/run_reconstruction.py`: inference pipeline producing per-frame PLY files.
- `scripts/finetune_vggt.py`: optional fine-tuning entrypoint.
- `src/vggt_reconstruction/*`: reusable dataset, model, geometry, training modules.

## Immediate next actions
1. Populate CSV manifests with real SoccerNet MVFoul paths.
2. Run reconstruction on 10 clips and inspect PLY quality.
3. If stable, generate pseudo-depth or GT depth files for fine-tuning.
4. Integrate generated point clouds into the foul classifier input pipeline.
