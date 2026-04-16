# Taller 2: Semantic Segmentation Pipeline
## TLDR
To run the interactive app, install [uv](https://docs.astral.sh/uv/getting-started/installation), unzip the dataset in the `data` dir, unzip the logs in the `logs` dir (ask the author), and then run the following:
- `make install`
- `make gradio`

This will launch an interactive app in `localhost:7860` that shows the metrics for each tried version in the leaderboard and allows inference for each track and frame. Currently the best version can be accessed at the checkpoint:
```
t2-t029-efficientnet-b2-dice/version_0/unet-timm-efficientnet-b2-dice-e09-miou0.6716.ckpt
```

In case the training curves are of interest, run:
```
make tensorboard
```

## 1. Introduction

This project solves a **7-class semantic segmentation** task for SuperTuxKart frames (400x400).
The objective is to classify each pixel into:

1. background
2. track
3. kart
4. pickup
5. nitro
6. bomb
7. projectile

The main challenge is **strong class imbalance**. Most pixels belong to background/track/kart, while pickup, nitro, bomb, and projectile are sparse. The implementation focuses on simple and readable techniques that improve minority-class learning without adding unnecessary complexity.

## 2. Project Structure

- computer_vision/taller_2/dataset.py: dataset scanning, filtering, augmentations, samplers, and LightningDataModule
- computer_vision/taller_2/model.py: LightningModule with SMP models, selectable backbones/losses, metrics, and TensorBoard image logging
- computer_vision/taller_2/train.py: single experiment entrypoint
- computer_vision/taller_2/grid_runner.py: experiment grid orchestration and leaderboard generation
- computer_vision/taller_2/visualization.py: Gradio app for qualitative inspection and leaderboard report browsing
- Makefile: reproducible command shortcuts

## 3. Data Assumptions

Dataset root: `data/dense_data`

Each track follows:

- `frame/frame_XXXX.png`
- `combined/mask_combined_XXXX.png`

Important filters:

- only frame ids `0000` to `0249` are used
- samples without combined masks are discarded
- all tracks are used in training/validation split
- validation uses a predefined fraction of samples and enforces minority-class presence ratio

## 4. Reproduce Results

### 4.1 Environment setup

```bash
make install
```

### 4.2 Start from clean logs

```bash
make clean-logs
```

### 4.3 Quick pipeline sanity check

```bash
make train-smoke
```

This generates:

- a trained checkpoint
- TensorBoard logs with metrics and validation visual examples

### 4.4 Run experiment grid (high-end workflow)

Default high-end command:

```bash
make grid-highend
```

You can also run with custom overrides:

```bash
make grid-highend GPU_IDS=0,1,2,3 PARALLEL_WORKERS=4 EXPERIMENT_PREFIX=t2 MAX_EPOCHS=35
```

### 4.5 Resume interrupted grid run

```bash
make grid-resume OUTPUT_DIR=logs/grid_runs/<run-folder>
```

### 4.6 Inspect ranked results

```bash
make leaderboard
```

### 4.7 Visual monitoring

TensorBoard:

```bash
make tensorboard
```

Gradio app:

```bash
make gradio
```

Gradio includes:

- **Inference tab**: frame/GT/prediction/overlay + compact experiment summary + grouped validation metrics
- **Leaderboard Report tab**: ranked model report from grid runs

## 5. Implementation Logic and Key Tricks

### 5.1 Sampling and imbalance handling

The pipeline combines three simple mechanisms:

1. **Minority-focused crop sampling**
   - During training, crops are often centered around pixels from classes 3-6.
   - Fallback to random crop if minority pixels are absent.
   - Effect: increases minority-class visibility in batches.

2. **WeightedRandomSampler at image level**
   - Images with higher minority pixel ratio are sampled more frequently.
   - Effect: more balanced exposure to informative frames.

3. **Class-weighted loss support**
   - Class weights are derived from inverse frequency (smoothed, normalized).
   - Used in cross-entropy based losses.
   - Effect: reduces dominance of frequent classes.

4. **Minority-aware validation split**
   - Validation is sampled from all tracks with configurable `val_fraction`.
   - A target ratio of minority-present samples (`val_minority_fraction`) is enforced.
   - Effect: validation remains representative while still stressing minority classes.

### 5.2 Models, backbones, and losses

Models are created through `segmentation_models_pytorch` (SMP) with configurable architecture and encoder.

- Architecture examples: `Unet`, `UnetPlusPlus`
- Encoder/backbone examples: `resnet18`, `resnet34`, `timm-efficientnet-b2`
- Loss options: `cross_entropy`, `focal`, `dice`, `ce_dice`

Why this design:

- SMP gives a stable, modular interface
- easy grid search over backbones and losses
- fast iteration without rewriting model code

### 5.3 Metrics and experiment tracking

Logged metrics include:

- mean IoU
- IoU per class
- mean class accuracy
- per-class accuracy
- pixel accuracy

These metrics are tracked for train/validation. Model selection is based on `val_mean_iou`.

Additional tracking tricks:

- fixed set of 5 validation samples logged every epoch to TensorBoard
- descriptive checkpoint names with architecture/backbone/loss and validation mIoU
- grid runner generates `results.csv`, `summary.json`, and `leaderboard.csv`

### 5.4 Validation-only protocol

This version intentionally removes the test split to maximize training data and simplify model comparison. The key validation controls are:

- `val_fraction`: controls how many frames are reserved for validation
- `val_minority_fraction`: controls how many validation samples contain minority classes

This keeps evaluation stable across runs while focusing on practical model selection.

## 6. Recommended Workflow

1. `make clean-logs`
2. `make train-smoke`
3. `make grid-highend` (or `make grid-local` if needed)
4. `make leaderboard`
5. `make tensorboard` and `make gradio`
6. pick best checkpoints and inspect qualitative behavior in Gradio

