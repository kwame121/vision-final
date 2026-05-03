# Synthetic cartography — satellite-to-map translation

**Kwame Adaboh** · ICS555 Computer Vision · Ashesi University (final project)

This repository implements conditional image synthesis from paired aerial *maps* tiles (Pix2Pix-style): a PatchGAN-conditioned generator with a compact four-stage encoder–decoder and U-Net-style skips, trained with least-squares adversarial objectives and strong L1 reconstruction (weight 100). Three variants are reproduced: a **baseline** with null skip connections, a **structural** U-Net with skips but no rotation penalty, and **proposed** skips plus a rotation self-consistency loss (weight 10) to encourage alignment under ninety-degree rotations. Training and reporting use mean absolute error, peak SNR, a pooled structural-similarity surrogate, and a rotation-gap diagnostic; held-out aggregates in the written report are computed in a single pass after checkpoints are frozen and are not used for model selection.

## Download artifacts

| Archive | Link |
| -------- | ---- |
| **datasets.rar** | [Google Drive — datasets.rar](https://drive.google.com/file/d/1f381FIqZZhXgn1Lk3E-1j5xdHqDmjTcJ/view?usp=drive_link) |
| **checkpoints.rar** | [Google Drive — checkpoints.rar](https://drive.google.com/file/d/1jW2hZFKNbgzN-OfNlayds8byuL-nlLYK/view?usp=drive_link) |

Extract **`datasets.rar`** at the **repository root** so it restores the **`datasets/`** folder (rename or backup an existing folder first). You should see `datasets/maps/maps/train/` (paired JPG/PNG, left satellite | right map), `datasets/maps/maps/val/` or `datasets/maps/maps/val_select/`, and optionally `datasets/maps/maps/heldout_test/`.

Extract **`checkpoints.rar`** at the **repository root** so it restores **`checkpoints/`**. The bundled long-run snapshots use explicit names: **`proposed_latest_epoch90.pt`** (best-looking ~90-epoch weights) and **`proposed_latest_epoch50.pt`** (mid-training reference). Routine training runs in this codebase also emit **`proposed_latest.pt`** / **`proposed_best.pt`**. For qualitative demos sharper than the ten-epoch matched ablations, prefer **`checkpoints/proposed_latest_epoch90.pt`** over **`checkpoints_exp_ablation/proposed_bs8/proposed_best.pt`**.

When present in the archive, **`checkpoints/unet_latest.pt`** and **`checkpoints/baseline_latest.pt`** are the long-run structural and baseline generators.

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

This reruns matched **baseline**, **unet**, and **proposed** into **`checkpoints_exp_ablation/{baseline_bs8,unet_bs8,proposed_bs8}/`** and **`results_exp_ablation/...`** only—it does **not** overwrite **`checkpoints/`** from **`checkpoints.rar`** unless you change `--checkpoints-dir`.

Optional pass-through arguments (appended to each run), e.g.:

```bash
python scripts/run_ablations.py --img-size 256
```

## Single-image inference

Long-run **`checkpoints/`** (recommended—use **`proposed_latest_epoch90.pt`** for best visuals):

```bash
python scripts/inference_single.py --input path/to/tile.png --resume-from checkpoints/proposed_latest_epoch90.pt --variant proposed --format satellite
```

Earlier archival snapshot (**`proposed_latest_epoch50.pt`**):

```bash
python scripts/inference_single.py --input path/to/tile.png --resume-from checkpoints/proposed_latest_epoch50.pt --variant proposed --format satellite
```

Matched ten-epoch ablation checkpoint (comparison with report table):

```bash
python scripts/inference_single.py --input path/to/tile.png --resume-from checkpoints_exp_ablation/proposed_bs8/proposed_best.pt --variant proposed --format satellite
```

Other variants from **`checkpoints/`** when shipped: **`checkpoints/unet_latest.pt`** with **`--variant unet`**, **`checkpoints/baseline_latest.pt`** with **`--variant baseline`**.

Use `--format paired` for the standard left-satellite / right-map training crop.

## Default single run (not ablation matrix)

```bash
python scripts/run_pipeline.py --mode train_eval --variant proposed --results-dir results --checkpoints-dir checkpoints
```

## Held-out split (optional)

See [scripts/EXPERIMENT_QUEUE_USAGE.md](scripts/EXPERIMENT_QUEUE_USAGE.md) for `create_heldout_split.py` and `materialize_split.py`.
