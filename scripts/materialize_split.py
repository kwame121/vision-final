"""Materialize split manifests into concrete directories via file copy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _copy_from_manifest(manifest_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    copied = 0
    for src in lines:
        src_path = Path(src)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing source file in manifest: {src_path}")
        shutil.copy2(src_path, out_dir / src_path.name)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy files from split manifests to concrete folders")
    parser.add_argument("--manifests-dir", default="results_exp/splits")
    parser.add_argument("--heldout-out", default="datasets/maps/maps/heldout_test")
    parser.add_argument("--val-select-out", default="datasets/maps/maps/val_select")
    args = parser.parse_args()

    manifests_dir = Path(args.manifests_dir)
    heldout_manifest = manifests_dir / "heldout_test.txt"
    val_manifest = manifests_dir / "val_select.txt"
    if not heldout_manifest.exists() or not val_manifest.exists():
        raise FileNotFoundError("Expected heldout_test.txt and val_select.txt in manifests directory")

    heldout_copied = _copy_from_manifest(heldout_manifest, Path(args.heldout_out))
    val_copied = _copy_from_manifest(val_manifest, Path(args.val_select_out))
    print(f"Copied {heldout_copied} held-out files and {val_copied} val-select files.")


if __name__ == "__main__":
    main()
