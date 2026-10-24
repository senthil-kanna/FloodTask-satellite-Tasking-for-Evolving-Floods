#!/usr/bin/env bash
# Second-machine training (Linux). Run from the unzipped bundle folder:  bash run_machine_b.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  .venv/bin/pip install numpy scipy pandas matplotlib pyyaml tqdm requests pytest rasterio
  .venv/bin/pip install -e .
fi
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest -q tests/test_env.py
mkdir -p results/logs
.venv/bin/python -u scripts/train_all.py --parallel 2 --machine b 2>&1 | tee -a results/logs/train_all.log
zip -qr results_machine_b.zip results/rl results/logs
echo "DONE. Send results_machine_b.zip back to the project owner."
