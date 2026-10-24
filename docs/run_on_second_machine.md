# Running half of the PPO training on a second machine

The 25 PPO training runs are independent. They are split between two machines in `configs/split.yaml`:
- **Machine A (project laptop):** 12 runs
- **Machine B (second machine):** 13 runs

All settings are frozen before the split, so both machines train identical configurations. Only the seeds differ.

## Requirements for machine B
- NVIDIA GPU with >= 4 GB memory and a recent driver (CUDA 12.x capable). CPU-only works but is ~3x slower.
- Windows 10/11 or Linux, Python 3.11 or 3.12 on PATH, ~3 GB free disk, internet for the first install.
- Expected time: ~6 h on an RTX 4050-class GPU (2 runs in parallel). Keep the machine plugged in with sleep disabled.

## Steps (machine B)
1. Unzip `floodtask_train_bundle.zip` (~0.1 GB). It contains the code, training/validation scenes and frozen
   calibration files. No dataset download is needed, and test-event data are not included.
2. Windows: open PowerShell in the folder and run
   `powershell -ExecutionPolicy Bypass -File run_machine_b.ps1`
   Linux: `bash run_machine_b.sh`
3. The script installs the environment (first time only), checks that the GPU is visible, runs the environment unit
   tests, then trains the 13 runs. Progress: `results/logs/train_all.log` and `results/logs/ppo_<variant>_seed<k>.log`.
4. **Resumable**: if the machine restarts, run the same command again. Finished runs are skipped. A run that was cut off
   midway restarts from scratch.
5. When it prints `DONE`, send back `results_machine_b.zip` (a few hundred MB at most).

## Merging (machine A)
```powershell
.\.venv\Scripts\python scripts\merge_machine_b.py path\to\results_machine_b.zip
```
The merge script copies `results/rl/<variant>/seed<k>/` folders that do not exist yet and checks each run's
`train_log.json`. It reports the GPU each run used (taken from the provenance record).

## Reproducibility note
GPU type and driver differ between machines, so the trained weights are not bit-identical to what machine A would
produce. Every run records its hardware in `train_log.json -> provenance`, and evaluation of all checkpoints
happens on machine A under identical conditions.
