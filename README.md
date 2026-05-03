# Synthetic cartography — satellite-to-map translation

**Kwame Adaboh** · ICS555 Computer Vision · Ashesi University (final project)

This repository implements conditional image synthesis from paired aerial *maps* tiles (Pix2Pix-style): a PatchGAN-conditioned generator with a compact four-stage encoder–decoder and U-Net-style skips, trained with least-squares adversarial objectives and strong L1 reconstruction (weight 100). Three variants are reproduced: a **baseline** with null skip connections, a **structural** U-Net with skips but no rotation penalty, and **proposed** skips plus a rotation self-consistency loss (weight 10) to encourage alignment under ninety-degree rotations. Training and reporting use mean absolute error, peak SNR, a pooled structural-similarity surrogate, and a rotation-gap diagnostic; held-out aggregates in the written report are computed in a single pass after checkpoints are frozen and are not used for model selection.

## Download artifacts

| Archive | Link |
| -------- | ---- |
| **datasets.rar** | [Google Drive — datasets.rar](https://drive.google.com/file/d/1f381FIqZZhXgn1Lk3E-1j5xdHqDmjTcJ/view?usp=drive_link) |
| **checkpoints.rar** | [Google Drive — checkpoints.rar](https://drive.google.com/file/d/1jW2hZFKNbgzN-OfNlayds8byuL-nlLYK/view?usp=drive_link) |

Extract **`datasets.rar`** so training images live under `datasets/maps/maps/train/` (paired JPG/PNG in the standard left-satellite | right-map layout). Validation should be in `datasets/maps/maps/val_select/` if you use the held-out workflow, otherwise `datasets/maps/maps/val/`. Optionally add `datasets/maps/maps/heldout_test/` for a final eval only.

Extract **`checkpoints.rar`** so that pretrained weights sit under `checkpoints_exp_ablation/baseline_bs8/`, `checkpoints_exp_ablation/unet_bs8/`, and `checkpoints_exp_ablation/proposed_bs8/` (or move the contained `*_best.pt` files to those paths). The commands below assume those locations.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

## Run all three matched ablations (10 epochs, batch 8)

From the repository root:

```bash
python scripts/run_ablations.py
```

Or:

```powershell
.\scripts\run_ablations.ps1
```

```bash
bash scripts/run_ablations.sh
```

This runs, in order, **baseline**, **unet** (structural), and **proposed**, writing to `checkpoints_exp_ablation/{baseline_bs8,unet_bs8,proposed_bs8}/` and `results_exp_ablation/...`.

Optional pass-through arguments (appended to each run), e.g.:

```bash
python scripts/run_ablations.py --img-size 256
```

## Single-image inference

```bash
python scripts/inference_single.py --input path/to/tile.png --resume-from checkpoints_exp_ablation/proposed_bs8/proposed_best.pt --variant proposed --format satellite
```

Use `--format paired` for the standard left-satellite / right-map training crop.

## Default single run (not ablation matrix)

```bash
python scripts/run_pipeline.py --mode train_eval --variant proposed --results-dir results --checkpoints-dir checkpoints
```

## Held-out split (optional)

See [scripts/EXPERIMENT_QUEUE_USAGE.md](scripts/EXPERIMENT_QUEUE_USAGE.md) for `create_heldout_split.py` and `materialize_split.py`.
