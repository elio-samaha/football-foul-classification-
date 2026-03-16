# Next Stage Handover (for teammate takeover)

This document explains exactly what was implemented, how to run it, and how to continue to the next project stage safely.

## A. What has been implemented

1. **Temporal clip input from SoccerNet MVFoul manifests**
   - Loader reads clip windows from CSV and samples frame sequences.
2. **Depth estimation backbone wrapper**
   - Uses pre-trained `Intel/dpt-hybrid-midas` through `transformers`.
   - Wrapper interface (`VGGTDepthModel`) is intentionally swappable.
3. **3D projection output**
   - Converts each frame depth map to `XYZRGB` point cloud.
   - Exports PLY files per frame.
4. **Optional fine-tuning**
   - Training loop supports sidecar `*_depth.npy` supervision.
   - Saves checkpoints every epoch.

## B. Exact setup to run from scratch

1. Clone repo and enter it.
2. Create environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies.

```bash
pip install -r requirements-dev.txt
```

4. Place SoccerNet-MVFoul frames/labels under `data/SoccerNet-MVFoul/...` as described in README.
5. Update CSV manifests in `configs/` with real clip paths.

## C. Run commands

### Reconstruction (pre-trained)

```bash
python scripts/run_reconstruction.py
```

### Fine-tuning (if depth targets exist)

```bash
python scripts/finetune_vggt.py
```

### Basic checks

```bash
python -m compileall src scripts tests
pytest -q
```

## D. What you still must do manually

1. **Download/prepare SoccerNet-MVFoul data** and extract frames.
2. **Provide/derive depth targets** (`*_depth.npy`) if you want fine-tuning.
3. **Tune manifests** to align clip windows with your label strategy.
4. **Set camera intrinsics** (`fx, fy, cx, cy`) in config for better geometric accuracy.

## E. Recommended next-stage tasks (priority order)

1. **Add temporal consistency losses**
   - Adjacent frame depth consistency.
   - Photometric reprojection loss where possible.
2. **Point-cloud fusion per clip**
   - Register and fuse frame-level clouds into sliding-window clouds.
3. **Feature extraction for classifier**
   - Build a point-cloud encoder (e.g., PointNet++ / sparse voxel model).
   - Export frame-level embeddings + timestamps.
4. **2D + 3D late fusion**
   - Keep RGB stream for action cues.
   - Fuse with 3D stream in temporal transformer.

## F. Suggested interface contract for downstream classifier

For each clip:

- `rgb_frames`: `[T, 3, H, W]`
- `point_cloud_paths`: list of `T` PLY files
- `foul_label`: class id / multi-label vector
- `metadata`: camera id, timestamps, split id

This contract keeps training pipelines modular and easier to debug.

## G. Common failure modes and fixes

1. **No frames found**
   - Verify `frames_dir` in CSV and image extensions (`.jpg`/`.png`).
2. **Depth model download errors**
   - Check internet and HuggingFace access.
3. **CUDA not available**
   - Pipeline will run on CPU if configured fallback is used (slower).
4. **Bad geometry scale**
   - Re-check intrinsics and frame resize settings.

## H. Ownership notes

- Stage delivered as a working scaffold with clear extension points.
- Next owner should focus on temporal consistency + fused 3D representation and classifier integration.
