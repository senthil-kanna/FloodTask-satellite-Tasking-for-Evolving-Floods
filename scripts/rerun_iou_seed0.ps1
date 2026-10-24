# Rerun recurrent_iou seed0 (first attempt crashed: KeyError 'reward_iou_delta') after Stage B finishes.
# Waits for the running train_all process to exit, then trains the single job.
param([int]$WaitPid = 0)
Set-Location $PSScriptRoot\..
if ($WaitPid -gt 0) { Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue }
"start $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\logs\rerun_iou_seed0.log
& .\.venv\Scripts\python.exe -u scripts\train_all.py --parallel 1 --jobs recurrent_iou:0 *>&1 | Out-File -Append -Encoding utf8 results\logs\train_all.log
"done $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\logs\rerun_iou_seed0.log
