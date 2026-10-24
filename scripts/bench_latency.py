"""Isolated CPU benchmark for requirement R8 (decision latency, model size, peak memory).

  python scripts/bench_latency.py
Runs each PPO variant's seed-0 checkpoint alone on one CPU thread over a few test episodes (no other jobs in
parallel, unlike run_experiments.py whose per-episode latencies include contention from parallel workers).
Writes results/latency_bench.json.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from floodtask.env.factory import VARIANTS, make_env, specs
from floodtask.eval.runner import RLPolicy
from floodtask.rl.model import ActorCritic
from floodtask.utils import provenance, RESULTS, save_json

N_EPISODES = 4


def bench(variant: str) -> dict:
    m = ActorCritic(recurrent=VARIANTS[variant]["recurrent"])
    m.load_state_dict(torch.load(RESULTS / "rl" / variant / "seed0" / "best.pt", map_location="cpu", weights_only=True))
    env = make_env("test", variant)
    pol = RLPolicy(m, "cpu", name=f"ppo_{variant}")
    lat = []
    for spec in specs("test", N_EPISODES):
        obs, done = env.reset(spec), False
        pol.reset(0)
        while not done:
            t0 = time.perf_counter()
            a = pol.act(obs)
            lat.append(time.perf_counter() - t0)
            obs, _, done, _ = env.step(a)
    lat = 1000 * np.array(lat[16:])   # drop the first pass as warm-up
    return {"n_decisions": int(lat.size), "latency_ms_mean": float(lat.mean()), "latency_ms_median": float(np.median(lat)),
            "latency_ms_p95": float(np.percentile(lat, 95)), "latency_ms_max": float(lat.max()),
            "plan_16_tiles_ms_mean": float(16 * lat.mean()), "n_params": int(sum(p.numel() for p in m.parameters()))}


def peak_memory_mb() -> float:
    """Peak resident memory of this process (Windows peak working set; ru_maxrss elsewhere)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2**20
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    torch.set_num_threads(1)
    out = {"threads": 1, "device": "cpu", "variants": {}}
    for v in ("recurrent", "feedforward"):
        out["variants"][v] = bench(v)
        print(v, {k: round(x, 2) if isinstance(x, float) else x for k, x in out["variants"][v].items()})
    out["peak_rss_mb"] = peak_memory_mb()
    out["provenance"] = provenance()
    print("peak memory MB", round(out["peak_rss_mb"], 1))
    save_json(out, RESULTS / "latency_bench.json")


if __name__ == "__main__":
    main()
