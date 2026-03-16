# football-foul-classification-

This repository now includes **Stage 4.2 / Stage 1** scaffolding for VGGT-style scene reconstruction on SoccerNet MVFoul data.

### What is implemented
- **MVFoul integration**:
  - `scripts/download_mvfoul.py` uses the official SoccerNet API to download the MVFoul dataset (see [`soccernet/sn-mvfoul` on GitHub](https://github.com/SoccerNet/sn-mvfoul)).
  - `scripts/prepare_mvfoul_frames.py` converts the raw videos into frame folders and writes a CSV manifest consumed by the reconstruction pipeline.
- **Reconstruction pipeline**:
  - `SoccerNetMVFoulDataset` for sequence-aware clip loading.
  - `ReconstructionPipeline` for per-frame depth estimation and point-cloud export.
- **Depth backbones (VGGT stage)**:
  - DPT-based transformer (`Intel/dpt-hybrid-midas`) as a practical VGGT proxy.
  - Optional **real VGGT backbone** via the official [`facebookresearch/vggt`](https://github.com/facebookresearch/vggt) implementation.
- **Training utilities**:
  - Optional fine-tuning loop using `*_depth.npy` supervision when available.

### Folder structure
- `src/vggt_reconstruction/`: core modules (dataset, model, geometry, training, pipeline).
- `scripts/run_reconstruction.py`: reconstruction entrypoint.
- `scripts/finetune_vggt.py`: fine-tuning entrypoint.
- `scripts/download_mvfoul.py`: SoccerNet MVFoul downloader.
- `scripts/prepare_mvfoul_frames.py`: converts SoccerNet videos into frame folders + CSV.
- `configs/*.csv`: example split manifests.
- `docs/stage_4_2_stage_1_vggt_plan.md`: implementation + fine-tuning plan.

### Environment setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Download the MVFoul dataset

First, request access to the SoccerNet-MVFoul dataset and obtain your NDA password, then run:

```bash
python scripts/download_mvfoul.py --password YOUR_SOCCERNET_PASSWORD
```

This downloads the MVFoul splits into `data/SoccerNet` using the official SoccerNet API as described in the SoccerNet MVFoul README (`https://github.com/SoccerNet/sn-mvfoul`).

You can also omit `--password` and set the `SOCCERNET_PASSWORD` environment variable instead.

### 2. Prepare frame folders and the reconstruction CSV

Once the videos are downloaded, convert a subset into frame folders and generate a manifest compatible with the `SoccerNetMVFoulDataset`:

```bash
python scripts/prepare_mvfoul_frames.py \
  --soccernet-root data/SoccerNet \
  --output-root data/SoccerNet-MVFoul \
  --split Train \
  --max-videos 50 \
  --max-frames-per-video 96 \
  --manifest-path configs/reconstruction_split.csv
```

This will create:
- Frame folders in `data/SoccerNet-MVFoul/frames/CLIP_ID`.
- A CSV at `configs/reconstruction_split.csv` with rows:

```text
clip_id,frames_dir,start_idx,num_frames,label_json_path
```

### 3. Run the reconstruction pipeline (DPT proxy backbone)

With frames and a manifest in place, you can run the reconstruction pipeline:

```bash
python scripts/run_reconstruction.py
```

By default this uses the DPT-based depth model (`backend="dpt"`, `model_name="Intel/dpt-hybrid-midas"`) and writes per-frame PLY point clouds under `outputs/reconstruction/CLIP_ID`.

### 4. Using a real VGGT model

To use a real VGGT backbone instead of the DPT proxy:

1. Ensure the official VGGT repo is installed (it is included in `requirements.txt` as a Git dependency). If needed, install manually:

   ```bash
   pip install git+https://github.com/facebookresearch/vggt.git
   ```

2. Edit `scripts/run_reconstruction.py` and set:

   ```python
   backend="vggt",
   model_name="VGGT_Base",  # or another checkpoint from the VGGT repo
   ```

3. Run:

   ```bash
   python scripts/run_reconstruction.py
   ```

The `VGGTDepthModel` wrapper will route calls to the real VGGT implementation while keeping the same `DepthOutput` interface expected by the rest of the pipeline.

### 5. Optional fine-tuning

If you have ground-truth or pseudo depth maps stored as `.npy` files next to each RGB frame:
- `frame_0001.jpg`
- `frame_0001_depth.npy`

You can fine-tune the depth backbone with:

```bash
python scripts/finetune_vggt.py
```

The fine-tuning configuration (`FinetuneConfig`) supports both DPT and VGGT backends via its `backend` field.

### Notes
- Start with pre-trained inference if depth supervision is not yet ready.
- Keep per-frame PLY outputs; they are useful for temporal 3D reasoning in foul classification.
