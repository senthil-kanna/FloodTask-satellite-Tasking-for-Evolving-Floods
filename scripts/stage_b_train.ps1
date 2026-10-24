# STAGE B (overnight): all PPO variants x 5 seeds, 3 in parallel. Resumable: finished runs are skipped.
# Run:  powershell -ExecutionPolicy Bypass -File scripts\stage_b_train.ps1
# Keep the laptop plugged in with sleep disabled. Progress: results\logs\train_all.log and results\logs\ppo_*.log
Set-Location $PSScriptRoot\..
& .\.venv\Scripts\python.exe -u scripts\train_all.py --parallel 2 --machine a *>&1 | Out-File -Append -Encoding utf8 results\logs\train_all.log
"done" | Out-File results\logs\stage_b_done.flag
