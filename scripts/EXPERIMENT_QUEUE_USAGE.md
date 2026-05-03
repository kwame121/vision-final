# Experiment Queue Usage

This queue is designed to avoid interference with active training by writing all artifacts to:

- `results_exp/`
- `checkpoints_exp/`

## 1) Optional: create held-out manifests

```powershell
python "c:\Users\kwame\vision-final\scripts\create_heldout_split.py" --source-dir "c:\Users\kwame\vision-final\datasets\maps\maps\val" --output-dir "c:\Users\kwame\vision-final\results_exp\splits" --heldout-ratio 0.2 --seed 42
```

## 2) Optional: materialize held-out and val-select directories

```powershell
python "c:\Users\kwame\vision-final\scripts\materialize_split.py" --manifests-dir "c:\Users\kwame\vision-final\results_exp\splits" --heldout-out "c:\Users\kwame\vision-final\datasets\maps\maps\heldout_test" --val-select-out "c:\Users\kwame\vision-final\datasets\maps\maps\val_select"
```

## 3) Run queue in foreground

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "c:\Users\kwame\vision-final\scripts\run_experiment_queue.ps1"
```

## 4) Run queue detached (recommended)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "c:\Users\kwame\vision-final\scripts\launch_experiment_queue_detached.ps1"
```

Equivalent one-liner:

```powershell
Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"c:\Users\kwame\vision-final\scripts\run_experiment_queue.ps1`" *> `"c:\Users\kwame\vision-final\results_exp\queue.log`" 2>&1" -WindowStyle Hidden
```

Task Scheduler fallback:

```powershell
schtasks /Create /SC ONCE /TN "VisionExperimentQueue" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File c:\Users\kwame\vision-final\scripts\run_experiment_queue.ps1" /ST 23:59 /F
schtasks /Run /TN "VisionExperimentQueue"
```

## 5) Monitor progress

```powershell
Get-Content "c:\Users\kwame\vision-final\results_exp\queue.log" -Wait
```

Run-level logs are written under `results_exp/logs/` and run status is tracked in `results_exp/run_ledger.csv`.

## 6) Held-out test workflow (late split)

If you did not fix a test split before training, generate manifests, materialize folders, then use **`val_select` only for validation during any new tuning**. Report metrics on **`heldout_test` once**, after choosing the final checkpoint (no tuning on held-out).

1. Split from your current validation folder (reproducible seed):

```powershell
python "c:\Users\kwame\vision-final\scripts\create_heldout_split.py" --source-dir "c:\Users\kwame\vision-final\datasets\maps\maps\val" --output-dir "c:\Users\kwame\vision-final\results_exp\splits" --heldout-ratio 0.2 --seed 42
```

2. Copy files into concrete directories (copies preserve originals):

```powershell
python "c:\Users\kwame\vision-final\scripts\materialize_split.py" --manifests-dir "c:\Users\kwame\vision-final\results_exp\splits" --heldout-out "c:\Users\kwame\vision-final\datasets\maps\maps\heldout_test" --val-select-out "c:\Users\kwame\vision-final\datasets\maps\maps\val_select"
```

3. **Training / checkpoint selection**: point validation at `val_select` (keep train path as usual):

```powershell
python "c:\Users\kwame\vision-final\scripts\run_pipeline.py" --mode train_eval --variant proposed --val-dir "c:\Users\kwame\vision-final\datasets\maps\maps\val_select" ...other args...
```

Optional override for train data:

```powershell
... --train-dir "c:\Users\kwame\vision-final\datasets\maps\maps\train" ...
```

4. **Final test evaluation (once)**: load best checkpoint and evaluate only on held-out:

```powershell
python "c:\Users\kwame\vision-final\scripts\run_pipeline.py" --mode eval --variant proposed --resume-from "PATH\TO\proposed_best.pt" --eval-dir "c:\Users\kwame\vision-final\datasets\maps\maps\heldout_test" --batch-size 4 --seed 42
```

Notes:

- Existing runs that trained on full `val` should be described honestly in the report (prior exposure of held-out proxies). Prefer re-selecting checkpoints using `val_select` only for strongest claims, or label held-out metrics as supplementary.
- `train_eval` without `--eval-dir` runs post-train eval on the same directory as `--val-dir` / default `val`. Do not pass `--eval-dir` to held-out until the final `--mode eval` step.
