"""Run pix2pix train/eval loop with ablations, metrics, and resume support."""

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torchvision.utils as vutils
from torch import nn
from tqdm import tqdm

from config import Config
from data import mapdata
from models.Pix2Pix import Pix2Pix


def set_seed(seed: int):
    """Set seed for reproducible training."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)


def _rotation_consistency_loss(model: Pix2Pix, x: torch.Tensor, criterion_l1: nn.Module) -> torch.Tensor:
    x_rot = torch.rot90(x, k=1, dims=(2, 3))
    fake_rot = model.generator(x_rot)
    fake = model.generator(x)
    return criterion_l1(fake_rot, torch.rot90(fake, k=1, dims=(2, 3)))


def _to_01_range(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp((tensor + 1.0) * 0.5, 0.0, 1.0)


def _compute_batch_metrics(fake: torch.Tensor, target: torch.Tensor, model: Pix2Pix, x: torch.Tensor) -> Dict[str, float]:
    fake_01 = _to_01_range(fake)
    target_01 = _to_01_range(target)
    mae = torch.mean(torch.abs(fake_01 - target_01)).item()
    mse = torch.mean((fake_01 - target_01) ** 2).item()
    psnr = 10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-8))).item()

    # Lightweight global SSIM approximation. SSIM is a measure of image similarity. It uses the mean, variance, and covariance of the images.
    mu_x = torch.mean(fake_01)
    mu_y = torch.mean(target_01)
    sigma_x = torch.var(fake_01, unbiased=False) #computing variance of the fake image
    sigma_y = torch.var(target_01, unbiased=False) #computing variance of the target image
    sigma_xy = torch.mean((fake_01 - mu_x) * (target_01 - mu_y))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = float(((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)))

    with torch.no_grad():
        fake_rot = model.generator(torch.rot90(x, k=1, dims=(2, 3)))
    rot_error = torch.mean(torch.abs(fake_rot - torch.rot90(fake, k=1, dims=(2, 3)))).item()
    return {"mae": mae, "psnr": psnr, "ssim": ssim, "rotation_error": rot_error}


def _compute_losses(
    outputs: dict,
    y: torch.Tensor,
    criterion_gan: nn.Module,
    criterion_l1: nn.Module,
    lambda_l1: float,
    geom_loss: Optional[torch.Tensor] = None,
    lambda_geom: float = 0.0,
):
    pred_real = outputs["real_d"]
    pred_fake_detached = outputs["fake_d_detached"]
    pred_fake_for_g = outputs["fake_d_for_g"]
    fake = outputs["fake"]

    real_targets = torch.ones_like(pred_real)
    fake_targets = torch.zeros_like(pred_fake_detached)

    loss_d_real = criterion_gan(pred_real, real_targets)
    loss_d_fake = criterion_gan(pred_fake_detached, fake_targets)
    loss_d = 0.5 * (loss_d_real + loss_d_fake)

    loss_g_gan = criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
    loss_g_l1 = criterion_l1(fake, y)
    loss_g = loss_g_gan + lambda_l1 * loss_g_l1
    loss_geom = torch.tensor(0.0, device=fake.device)
    if geom_loss is not None and lambda_geom > 0: # for our ablative studies, currently set to default lol
        loss_geom = geom_loss
        loss_g = loss_g + lambda_geom * loss_geom

    return {
        "loss_d": loss_d,
        "loss_g": loss_g,
        "loss_g_gan": loss_g_gan,
        "loss_g_l1": loss_g_l1,
        "loss_geom": loss_geom,
    }


def evaluate(
    model: Pix2Pix,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion_gan: nn.Module,
    criterion_l1: nn.Module,
    lambda_l1: float,
    use_geom_loss: bool,
    lambda_geom: float,
):
    model.eval()
    totals = {
        "loss_d": 0.0,
        "loss_g": 0.0,
        "loss_g_gan": 0.0,
        "loss_g_l1": 0.0,
        "loss_geom": 0.0,
        "mae": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "rotation_error": 0.0,
    }
    n_batches = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x = x.to(device=device, non_blocking=True)
            y = y.to(device=device, non_blocking=True)
            outputs = model(x, y)
            geom = None
            if use_geom_loss:
                x_rot = torch.rot90(x, k=1, dims=(2, 3))
                fake_rot = model.generator(x_rot)
                geom = criterion_l1(fake_rot, torch.rot90(outputs["fake"], k=1, dims=(2, 3)))
            losses = _compute_losses(outputs, y, criterion_gan, criterion_l1, lambda_l1, geom, lambda_geom)
            metrics = _compute_batch_metrics(outputs["fake"], y, model, x)
            for key in ("loss_d", "loss_g", "loss_g_gan", "loss_g_l1", "loss_geom"):
                totals[key] += losses[key].item()
            for key in ("mae", "psnr", "ssim", "rotation_error"):
                totals[key] += metrics[key]
            n_batches += 1

    if n_batches == 0:
        return totals
    return {key: value / n_batches for key, value in totals.items()}


def train_epoch(
    model: Pix2Pix,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion_gan: nn.Module,
    criterion_l1: nn.Module,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    lambda_l1: float,
    use_geom_loss: bool,
    lambda_geom: float,
    grad_clip: Optional[float],
):
    model.train()
    totals = {
        "loss_d": 0.0,
        "loss_g": 0.0,
        "loss_g_gan": 0.0,
        "loss_g_l1": 0.0,
        "loss_geom": 0.0,
    }
    n_batches = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device=device, non_blocking=True)
        y = y.to(device=device, non_blocking=True)

        outputs_d = model(x, y)
        losses_d = _compute_losses(outputs_d, y, criterion_gan, criterion_l1, lambda_l1)
        optimizer_d.zero_grad(set_to_none=True)
        losses_d["loss_d"].backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.discriminator.parameters(), grad_clip)
        optimizer_d.step()

        outputs_g = model(x, y)
        geom = None
        if use_geom_loss:
            geom = criterion_l1(model.generator(torch.rot90(x, k=1, dims=(2, 3))), torch.rot90(outputs_g["fake"], k=1, dims=(2, 3)))
        losses_g = _compute_losses(outputs_g, y, criterion_gan, criterion_l1, lambda_l1, geom, lambda_geom)
        optimizer_g.zero_grad(set_to_none=True)
        losses_g["loss_g"].backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.generator.parameters(), grad_clip)
        optimizer_g.step()

        totals["loss_d"] += losses_d["loss_d"].item()
        totals["loss_g"] += losses_g["loss_g"].item()
        totals["loss_g_gan"] += losses_g["loss_g_gan"].item()
        totals["loss_g_l1"] += losses_g["loss_g_l1"].item()
        totals["loss_geom"] += losses_g["loss_geom"].item()
        n_batches += 1

    if n_batches == 0:
        return totals
    return {key: value / n_batches for key, value in totals.items()}


def save_checkpoint(
    checkpoint_path: Path,
    model: Pix2Pix,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val: float,
    config: Config,
):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
        "run_name": config.train.run_name,
        "variant": config.train.variant,
        "generator": model.generator.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def maybe_load_checkpoint(
    model: Pix2Pix,
    optimizer_d: torch.optim.Optimizer,
    optimizer_g: torch.optim.Optimizer,
    resume_from: Optional[str],
    device: torch.device,
):
    if not resume_from:
        return 0, 0, float("inf")
    ckpt_path = Path(resume_from)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.generator.load_state_dict(checkpoint["generator"])
    model.discriminator.load_state_dict(checkpoint["discriminator"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val = float(checkpoint.get("best_val", float("inf")))
    return start_epoch, global_step, best_val


def _append_metrics_files(config: Config, record: Dict):
    config.train.metrics_jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with config.train.metrics_jsonl_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    write_header = not config.train.metrics_csv_file.exists()
    with config.train.metrics_csv_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _save_eval_samples(model: Pix2Pix, loader: torch.utils.data.DataLoader, device: torch.device, config: Config, epoch: int):
    config.train.samples_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x = x.to(device)
        y = y.to(device)
        fake = model.generator(x)
        grid = torch.cat([_to_01_range(x[:1]), _to_01_range(y[:1]), _to_01_range(fake[:1])], dim=0)
        vutils.save_image(grid, config.train.samples_dir / f"epoch_{epoch + 1:03d}.png", nrow=3)


def run_training(config: Config):
    set_seed(seed=config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = mapdata.build_train_and_val_dataloaders(config=config)
    model = Pix2Pix(cfg=config).to(device)

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    d_params, g_params = model.get_param_groups()
    optimizer_d = torch.optim.Adam(d_params, lr=config.train.learning_rate, betas=(config.train.beta1, config.train.beta2), weight_decay=config.train.weight_decay)
    optimizer_g = torch.optim.Adam(g_params, lr=config.train.learning_rate, betas=(config.train.beta1, config.train.beta2), weight_decay=config.train.weight_decay)

    config.train.results_dir.mkdir(parents=True, exist_ok=True)
    config.train.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config.train.epoch_logs_file.parent.mkdir(parents=True, exist_ok=True)
    latest_path = config.train.checkpoints_dir / f"{config.train.variant}_latest.pt"
    best_path = config.train.checkpoints_dir / f"{config.train.variant}_best.pt"

    start_epoch, global_step, best_val = maybe_load_checkpoint(model, optimizer_d, optimizer_g, config.train.resume_from, device)
    best_epoch = start_epoch - 1
    best_metrics: Dict[str, float] = {}

    for epoch in range(start_epoch, config.train.num_epochs):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion_gan=criterion_gan,
            criterion_l1=criterion_l1,
            optimizer_d=optimizer_d,
            optimizer_g=optimizer_g,
            lambda_l1=config.train.lambda_l1,
            use_geom_loss=config.train.use_geom_loss,
            lambda_geom=config.train.lambda_geom,
            grad_clip=config.train.grad_clip,
        )
        global_step += len(train_loader)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion_gan=criterion_gan,
            criterion_l1=criterion_l1,
            lambda_l1=config.train.lambda_l1,
            use_geom_loss=config.train.use_geom_loss,
            lambda_geom=config.train.lambda_geom,
        )

        save_checkpoint(latest_path, model, optimizer_d, optimizer_g, epoch, global_step, best_val, config)
        if val_metrics["loss_g"] < best_val:
            best_val = val_metrics["loss_g"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(best_path, model, optimizer_d, optimizer_g, epoch, global_step, best_val, config)

        record = {
            "epoch": epoch + 1,
            "variant": config.train.variant,
            "train_loss_d": train_metrics["loss_d"],
            "train_loss_g": train_metrics["loss_g"],
            "train_loss_g_gan": train_metrics["loss_g_gan"],
            "train_loss_g_l1": train_metrics["loss_g_l1"],
            "train_loss_geom": train_metrics["loss_geom"],
            "val_loss_d": val_metrics["loss_d"],
            "val_loss_g": val_metrics["loss_g"],
            "val_loss_g_gan": val_metrics["loss_g_gan"],
            "val_loss_g_l1": val_metrics["loss_g_l1"],
            "val_loss_geom": val_metrics["loss_geom"],
            "val_mae": val_metrics["mae"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_rotation_error": val_metrics["rotation_error"],
        }
        _append_metrics_files(config, record)
        _save_eval_samples(model, val_loader, device, config, epoch)
        with config.train.epoch_logs_file.open("a", encoding="utf-8") as logf:
            logf.write(json.dumps(record) + "\n")

    summary = {
        "run_name": config.train.run_name,
        "variant": config.train.variant,
        "best_epoch": best_epoch + 1 if best_epoch >= 0 else None,
        "best_val_loss_g": best_val,
        "best_metrics": best_metrics,
        "checkpoints": {"latest": str(latest_path), "best": str(best_path)},
        "config": {
            "learning_rate": config.train.learning_rate,
            "lambda_l1": config.train.lambda_l1,
            "lambda_geom": config.train.lambda_geom,
            "use_geom_loss": config.train.use_geom_loss,
            "use_skip_connections": config.train.use_skip_connections,
            "batch_size": config.data.batch_size,
            "img_size": config.data.img_size,
        },
    }
    config.train.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with config.train.summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_evaluation(config: Config):
    set_seed(seed=config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = mapdata.build_eval_dataloader(config=config)
    model = Pix2Pix(cfg=config).to(device)
    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    d_params, g_params = model.get_param_groups()
    optimizer_d = torch.optim.Adam(d_params, lr=config.train.learning_rate, betas=(config.train.beta1, config.train.beta2))
    optimizer_g = torch.optim.Adam(g_params, lr=config.train.learning_rate, betas=(config.train.beta1, config.train.beta2))
    maybe_load_checkpoint(model, optimizer_d, optimizer_g, config.train.resume_from, device)
    metrics = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        criterion_gan=criterion_gan,
        criterion_l1=criterion_l1,
        lambda_l1=config.train.lambda_l1,
        use_geom_loss=config.train.use_geom_loss,
        lambda_geom=config.train.lambda_geom,
    )
    return metrics





