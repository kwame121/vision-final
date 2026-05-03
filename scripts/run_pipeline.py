"""Runs both training and eval for our pix2pix model implementation"""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config
from train.train import run_evaluation, run_training


def _apply_variant(config: Config, variant: str):
    if variant == "baseline":
        config.train.use_skip_connections = False
        config.train.use_geom_loss = False
        config.train.lambda_geom = 0.0
    elif variant == "unet":
        config.train.use_skip_connections = True
        config.train.use_geom_loss = False
        config.train.lambda_geom = 0.0
    elif variant == "proposed":
        config.train.use_skip_connections = True
        config.train.use_geom_loss = True
    else:
        raise ValueError(f"Unknown variant: {variant}")
    config.train.variant = variant


def run_pipeline(args):
    config = Config()
    config.train.mode = args.mode
    config.train.variant = args.variant
    config.train.run_name = args.run_name
    config.train.num_epochs = args.num_epochs
    config.train.learning_rate = args.lr
    config.train.lambda_l1 = args.lambda_l1
    config.train.lambda_geom = args.lambda_geom
    config.train.resume_from = args.resume_from
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    config.data.img_size = args.img_size
    config.data.eval_dir = args.eval_dir
    if args.train_dir is not None:
        config.data.train_dir = args.train_dir
    if args.val_dir is not None:
        config.data.val_dir = args.val_dir
    config.data.seed = args.seed
    config.train.seed = args.seed
    config.train.results_dir = Path(args.results_dir)
    config.train.checkpoints_dir = Path(args.checkpoints_dir)
    config.train.epoch_logs_file = config.train.results_dir / "epoch_logs.txt"
    config.train.metrics_jsonl_file = config.train.results_dir / "metrics.jsonl"
    config.train.metrics_csv_file = config.train.results_dir / "metrics.csv"
    config.train.summary_file = config.train.results_dir / "summary.json"
    config.train.samples_dir = config.train.results_dir / "samples"

    _apply_variant(config, args.variant)
    if args.mode in ("train", "train_eval"):
        summary = run_training(config)
        print(json.dumps(summary, indent=2))
    if args.mode in ("eval", "train_eval"):
        metrics = run_evaluation(config)
        print(json.dumps({"eval_metrics": metrics}, indent=2))


def main():
    parser = ArgumentParser(description="Pix2Pix experiment pipeline")
    parser.add_argument("--mode", choices=["train", "eval", "train_eval"], default="train_eval")
    parser.add_argument("--variant", choices=["baseline", "unet", "proposed"], default="proposed")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--run-name", default="pix2pix_experiment")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval-dir",
        default=None,
        help="Directory used by eval (--mode eval or final eval after train_eval). Defaults to --val-dir / config val_dir.",
    )
    parser.add_argument("--train-dir", default=None, help="Override train image directory")
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Override validation directory (after split, point to datasets/.../val_select)",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--lambda-geom", type=float, default=10.0)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "checkpoints"))
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()