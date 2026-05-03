"""Create deterministic validation-selection and held-out test file lists.

This script does not modify source images. It writes text manifests that can be
used to construct explicit selection/test subsets without contaminating training.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _list_images(src_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in src_dir.iterdir() if p.suffix.lower() in exts and p.is_file()])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create held-out split manifests from a source directory")
    parser.add_argument("--source-dir", required=True, help="Directory containing candidate evaluation images")
    parser.add_argument("--output-dir", default="results_exp/splits", help="Directory to store split manifests")
    parser.add_argument("--heldout-ratio", type=float, default=0.2, help="Fraction assigned to held-out test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _list_images(source_dir)
    if not files:
        raise FileNotFoundError(f"No image files found in {source_dir}")

    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)

    heldout_count = max(1, int(len(shuffled) * args.heldout_ratio))
    heldout = sorted(shuffled[:heldout_count])
    val_select = sorted(shuffled[heldout_count:])

    (output_dir / "heldout_test.txt").write_text("\n".join(str(p) for p in heldout) + "\n", encoding="utf-8")
    (output_dir / "val_select.txt").write_text("\n".join(str(p) for p in val_select) + "\n", encoding="utf-8")
    (output_dir / "split_summary.txt").write_text(
        "\n".join(
            [
                f"source_dir={source_dir}",
                f"seed={args.seed}",
                f"total={len(shuffled)}",
                f"heldout_count={len(heldout)}",
                f"val_select_count={len(val_select)}",
                f"heldout_ratio={args.heldout_ratio}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote split manifests to {output_dir}")


if __name__ == "__main__":
    main()
