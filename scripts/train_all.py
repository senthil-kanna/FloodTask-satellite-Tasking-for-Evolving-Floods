"""Train all PPO variants x seeds (skips runs whose train_log.json exists).
  python scripts/train_all.py --parallel 2 [--variants recurrent feedforward] [--seeds 0 1 2 3 4]
  python scripts/train_all.py --parallel 2 --machine a      # this machine's half (configs/split.yaml)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from floodtask.env.factory import VARIANTS
from floodtask.utils import RESULTS, ROOT


def run(job):
    variant, seed = job
    if (RESULTS / "rl" / variant / f"seed{seed}" / "train_log.json").exists():
        return f"{variant} seed{seed}: exists"
    log = RESULTS / "logs" / f"ppo_{variant}_seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, "-u", "-m", "floodtask.rl.train_ppo", "--seed", str(seed), "--variant", variant],
                             cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return f"{variant} seed{seed}: exit {rc}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=2)   # >2 CUDA processes time-slice badly on one laptop GPU
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--jobs", nargs="+", default=None, help="explicit list variant:seed (overrides --variants/--seeds)")
    ap.add_argument("--machine", choices=["a", "b"], default=None, help="use the job list in configs/split.yaml")
    a = ap.parse_args()
    if a.machine:
        from floodtask.utils import load_yaml
        a.jobs = load_yaml(ROOT / "configs" / "split.yaml")[f"machine_{a.machine}"]
    if a.jobs:
        jobs = [(j.split(":")[0], int(j.split(":")[1])) for j in a.jobs]
    else:
        jobs = [(v, s) for s in a.seeds for v in a.variants]   # seed-major: every variant gets early seeds first
    with ThreadPoolExecutor(a.parallel) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)
