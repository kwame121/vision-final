from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing"""
    train_dir: str = "datasets/maps/maps/train"
    val_dir: str = "datasets/maps/maps/val"
    eval_dir: Optional[str] = None
    img_size: int = 256
    batch_size: int = 4
    num_workers: int = 2
    val_split: float = 0.1
    seed: int = 42
    

@dataclass
class TrainConfig:
    "Configuration for the training loop of the model"
    num_epochs: int = 10
    learning_rate: float = 0.0001
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    save_every: int = 1
    log_every: int = 100
    resume_from: Optional[str] = None
    grad_accumulation_steps: int = 1
    lambda_l1: float = 100.0
    lambda_geom: float = 10.0
    use_geom_loss: bool = True
    use_skip_connections: bool = True
    variant: str = "proposed"
    mode: str = "train_eval"
    run_name: str = field(default_factory=lambda: f"pix2pix_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    results_dir: Path = Path("results")
    checkpoints_dir: Path = Path("checkpoints")
    epoch_logs_file: Path = Path("results/epoch_logs.txt")
    metrics_jsonl_file: Path = Path("results/metrics.jsonl")
    metrics_csv_file: Path = Path("results/metrics.csv")
    summary_file: Path = Path("results/summary.json")
    samples_dir: Path = Path("results/samples")
    seed: int = 42



@dataclass
class Config:
    """Main configuration for the pix2pix model"""
    data:DataConfig = field(default_factory=DataConfig)
    train:TrainConfig = field(default_factory=TrainConfig)