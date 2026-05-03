#!/usr/bin/env bash
# Run matched ablations from repo root (baseline_bs8, unet_bs8, proposed_bs8).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python scripts/run_ablations.py "$@"
