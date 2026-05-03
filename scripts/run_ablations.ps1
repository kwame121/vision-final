# Run matched ablations from repo root (baseline_bs8, unet_bs8, proposed_bs8).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python "$PSScriptRoot\run_ablations.py" @args
