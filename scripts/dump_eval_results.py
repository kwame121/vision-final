"""Dump qualitative eval triptychs (satellite | prediction | roadmap) to disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from config import Config  # noqa: E402
from data.mapdata import MapDataset  # noqa: E402
from models.Pix2Pix import Pix2Pix  # noqa: E402
from run_pipeline import _apply_variant  # noqa: E402
from train.train import (  # noqa: E402
    _compute_batch_metrics,
    _to_01_range,
    maybe_load_checkpoint,
    set_seed,
)


def _build_config(args: argparse.Namespace) -> Config:
    config = Config()
    config.train.resume_from = args.resume_from
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.img_size = args.img_size
    config.data.seed = args.seed
    config.train.seed = args.seed
    config.data.eval_dir = str(Path(args.eval_dir).resolve())
    config.data.val_dir = config.data.eval_dir
    config.train.learning_rate = args.lr
    config.train.lambda_l1 = args.lambda_l1
    config.train.lambda_geom = args.lambda_geom
    config.train.weight_decay = args.weight_decay
    _apply_variant(config, args.variant)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Save N eval triptychs (input|pred|GT) as PNGs.")
    parser.add_argument("--resume-from", required=True, help="Checkpoint .pt path")
    parser.add_argument("--variant", choices=["baseline", "unet", "proposed"], default="proposed")
    parser.add_argument(
        "--eval-dir",
        default=str(ROOT / "datasets" / "maps" / "maps" / "val"),
        help="Directory with paired sat|map JPG/PNG (maps val split)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output folder (will contain images/ and manifest.json)",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--lambda-geom", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip per-image mae/psnr/ssim/rotation_error in manifest",
    )
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    config = _build_config(args)
    ckpt_path = Path(args.resume_from).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    eval_dir_resolved = Path(args.eval_dir).resolve()
    dataset = MapDataset(str(eval_dir_resolved), False, config.data)
    n_ds = len(dataset)
    if n_ds == 0:
        raise FileNotFoundError(f"No JPG/PNG in eval dir: {eval_dir_resolved}")

    num_to_write = min(args.num_samples, n_ds)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    set_seed(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Pix2Pix(cfg=config).to(device)
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    d_params, g_params = model.get_param_groups()
    optimizer_d = torch.optim.Adam(
        d_params,
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
    )
    optimizer_g = torch.optim.Adam(
        g_params,
        lr=config.train.learning_rate,
        betas=(config.train.beta1, config.train.beta2),
        weight_decay=config.train.weight_decay,
    )
    maybe_load_checkpoint(model, optimizer_d, optimizer_g, config.train.resume_from, device)
    model.eval()

    out_root = Path(args.out_dir).resolve()
    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict] = []
    written = 0
    global_row = 0

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            if written >= num_to_write:
                break
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            fake = model.generator(batch_x)
            bs = batch_x.shape[0]
            for i in range(bs):
                if written >= num_to_write:
                    break
                idx = global_row + i
                src_path = Path(dataset.image_paths[idx])
                xi = batch_x[i : i + 1]
                fi = fake[i : i + 1]
                yi = batch_y[i : i + 1]
                grid = torch.cat([_to_01_range(xi), _to_01_range(fi), _to_01_range(yi)], dim=0)
                fname = f"{written:04d}_{src_path.stem}.png"
                rel_output = f"images/{fname}"
                vutils.save_image(grid, str(images_dir / fname), nrow=3)
                entry: dict = {
                    "index": written,
                    "source": src_path.name,
                    "output": rel_output,
                }
                if not args.skip_metrics:
                    entry["metrics"] = _compute_batch_metrics(fi, yi, model, xi)
                manifest_entries.append(entry)
                written += 1
            global_row += bs

    manifest = {
        "checkpoint": str(ckpt_path),
        "variant": args.variant,
        "eval_dir": str(eval_dir_resolved),
        "num_written": written,
        "num_requested": args.num_samples,
        "samples": manifest_entries,
    }
    man_path = out_root / "manifest.json"
    with man_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {written} triptychs to {images_dir}")
    print(f"Manifest: {man_path}")


if __name__ == "__main__":
    main()
