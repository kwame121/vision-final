# Vision final — Pix2Pix satellite-to-map

Lightweight clone: **code + empty directory layout** live in git; **datasets**, **weights**, and **run logs** stay local (or come from your Drive / RAR restore).

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

## Restore data and checkpoints (local only)

1. Unpack your archives so **training tiles** sit under:

   `datasets/maps/maps/train/` (JPG/PNG pairs as in the Pix2Pix *maps* layout)

2. Put **validation** tiles under either:

   - `datasets/maps/maps/val_select/` (preferred if you materialized a held-out split), or  
   - `datasets/maps/maps/val/`

3. Optional: `datasets/maps/maps/heldout_test/` for a one-shot final eval after training (see [scripts/EXPERIMENT_QUEUE_USAGE.md](scripts/EXPERIMENT_QUEUE_USAGE.md)).

**Git** ignores image and checkpoint binaries under those trees; only `.gitkeep` placeholders define the folder layout.

## Run all three matched ablations (10 epochs, batch 8)

From the **repository root**:

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

This runs, in order, **baseline**, **unet** (structural), and **proposed** into:

- `checkpoints_exp_ablation/baseline_bs8/`, `unet_bs8/`, `proposed_bs8/`
- `results_exp_ablation/baseline_bs8/`, …

Pass-through args are supported (appended to each inner command), e.g.:

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

## What is excluded from this repo

- `serve/` — optional HTTP tile server (not required for training or `inference_single.py`).
- `report-draft/` — LaTeX draft.
- `results_dump/`, `results_exp/`, `checkpoints_exp/` — bulky experiment sandboxes.
- `results/` and `checkpoints/` — default artifact dirs (kept empty in git via `.gitkeep`).

## Held-out split (optional)

See [scripts/EXPERIMENT_QUEUE_USAGE.md](scripts/EXPERIMENT_QUEUE_USAGE.md) for `create_heldout_split.py` and `materialize_split.py`.
