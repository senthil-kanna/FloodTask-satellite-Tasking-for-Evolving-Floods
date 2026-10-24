# Detached PPO training launcher; all arguments are forwarded to scripts/train_all.py.
# Example: Start-Process powershell -ArgumentList '-NoProfile','-File','scripts\train_all.ps1','--parallel','2','--seeds','0'
Set-Location $PSScriptRoot\..
& .\.venv\Scripts\python.exe -u scripts\train_all.py @args *>&1 | Out-File -Append -Encoding utf8 results\logs\train_all.log
