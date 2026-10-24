# Second-machine training (Windows). Run from the unzipped bundle folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File run_machine_b.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) {
    python -m venv .venv
    .\.venv\Scripts\python -m pip install --upgrade pip
    .\.venv\Scripts\pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    .\.venv\Scripts\pip install numpy scipy pandas matplotlib pyyaml tqdm requests pytest rasterio
    .\.venv\Scripts\pip install -e .
}
.\.venv\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible - stop and tell the project owner'; print('GPU:', torch.cuda.get_device_name(0))"
.\.venv\Scripts\python -m pytest -q tests\test_env.py
New-Item -ItemType Directory -Force results\logs | Out-Null
.\.venv\Scripts\python -u scripts\train_all.py --parallel 2 --machine b *>&1 | Tee-Object -Append results\logs\train_all.log
Compress-Archive -Force -Path results\rl, results\logs -DestinationPath results_machine_b.zip
Write-Host "DONE. Send results_machine_b.zip back to the project owner."
