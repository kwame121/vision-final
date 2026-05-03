"""Run the three matched batch-size-8 ablations (baseline, structural/unet, proposed).

Requires `datasets/maps/maps/train` with images and `val` or `val_select` for validation.
Checkpoint and log directories are created automatically.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _val_dir() -> Path:
    val_select = ROOT / "datasets" / "maps" / "maps" / "val_select"
    val_default = ROOT / "datasets" / "maps" / "maps" / "val"
    if any(val_select.glob("*.jpg")) or any(val_select.glob("*.png")):
        return val_select
    return val_default


def _run(variant: str, run_suffix: str, extra_args: list[str]) -> None:
    train_dir = ROOT / "datasets" / "maps" / "maps" / "train"
    if not (any(train_dir.glob("*.jpg")) or any(train_dir.glob("*.png"))):
        print(f"ERROR: No training images under {train_dir}", file=sys.stderr)
        sys.exit(1)
    val_dir = _val_dir()
    if not (any(val_dir.glob("*.jpg")) or any(val_dir.glob("*.png"))):
        print(f"ERROR: No validation images under {val_dir}", file=sys.stderr)
        sys.exit(1)

    results_dir = ROOT / "results_exp_ablation" / run_suffix
    ckpt_dir = ROOT / "checkpoints_exp_ablation" / run_suffix
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--mode",
        "train_eval",
        "--variant",
        variant,
        "--run-name",
        run_suffix,
        "--num-epochs",
        "10",
        "--batch-size",
        "8",
        "--seed",
        "42",
        "--lr",
        "2e-4",
        "--lambda-l1",
        "100",
        "--lambda-geom",
        "10",
        "--train-dir",
        str(train_dir),
        "--val-dir",
        str(val_dir),
        "--results-dir",
        str(results_dir),
        "--checkpoints-dir",
        str(ckpt_dir),
        "--num-workers",
        "2",
        *extra_args,
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    extra = sys.argv[1:]
    _run("baseline", "baseline_bs8", extra)
    _run("unet", "unet_bs8", extra)
    _run("proposed", "proposed_bs8", extra)
    print("All three ablations finished.", flush=True)


if __name__ == "__main__":
    main()
