"""Run the predeclared experiment programme (docs/experimental_protocol.md) on held-out TEST scenes.

  python scripts/run_experiments.py --suite core|ablation|forecast|budget|revisit|latency|sensor|all
Raw per-episode outputs (per-pass metrics + full tasking audit trail) are written to results/experiments/<suite>/.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from floodtask.env.factory import make_env, specs
from floodtask.eval.runner import RLPolicy, run_episode
from floodtask.policies.baselines import BASELINES
from floodtask.rl.model import ActorCritic
from floodtask.utils import load_json, provenance, RESULTS, save_json

EP_PER_SCENE = 4
RL_SEEDS = [0, 1, 2, 3, 4]
OUT = RESULTS / "experiments"


def tuned(name):
    path = RESULTS / "baseline_params.json"
    if name in ("greedy_unc", "forecast", "flooded_first") and path.exists():
        return BASELINES[name](**load_json(path)[name]["params"])
    return BASELINES[name]()


def n_test_specs():
    from floodtask.env.factory import scene_names
    return EP_PER_SCENE * len(scene_names("test"))


def _baseline_job(args):
    name, env_kw, chunk = args
    env = make_env("test", **env_kw)
    pol = tuned(name)
    return [_strip(run_episode(env, pol, s), name, -1) for s in chunk]


def _rl_job(args):
    variant, seed, env_kw, chunk, label = args
    torch.set_num_threads(1)
    from floodtask.env.factory import VARIANTS
    m = ActorCritic(recurrent=VARIANTS[variant]["recurrent"])
    m.load_state_dict(torch.load(RESULTS / "rl" / variant / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True))
    env = make_env("test", variant, **env_kw)
    pol = RLPolicy(m, "cpu", name=label or f"ppo_{variant}")
    return [_strip(run_episode(env, pol, s), pol.name, seed) for s in chunk]


def _job(args):
    return _baseline_job(args[1:]) if args[0] == "b" else _rl_job(args[1:])


def _strip(ep, name, seed):
    ep["policy"], ep["train_seed"] = name, seed
    return ep


def run_suite(tag, baselines, rl_variants, env_kw=None, spec_kw=None, workers=12, n_chunks=12):
    env_kw, spec_kw = env_kw or {}, {"n": n_test_specs(), **(spec_kw or {})}
    if (OUT / tag / "meta.json").exists():  # finished on an earlier run
        print(f"{tag}: already done, skipped")
        return
    # every episode is seeded by its own spec (env) and policy_seed=0, so chunking does not change results;
    # one queue of small jobs keeps all workers busy; map() keeps the original episode order
    sp = specs("test", **spec_kw)
    chunks = [sp[i * len(sp) // n_chunks:(i + 1) * len(sp) // n_chunks] for i in range(n_chunks)]
    jobs_r = [(v, s) for v in rl_variants for s in RL_SEEDS if (RESULTS / "rl" / v / f"seed{s}" / "best.pt").exists()]
    jobs = [("b", b, env_kw, c) for b in baselines for c in chunks] + \
           [("r", v, s, env_kw, c, None) for v, s in jobs_r for c in chunks]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs))
    eps = [e for r in res for e in r]
    if tag != "core":  # full per-decision audit trail retained for the core suite only (file size)
        for e in eps:
            e.pop("history")
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    save_json({"tag": tag, "env_kw": env_kw, "spec_kw": spec_kw, "baselines": baselines, "rl_variants": rl_variants,
               "rl_seeds_found": sorted({(j[0], j[1]) for j in jobs_r}), "n_episodes": len(eps),
               "provenance": provenance()}, d / "meta.json")
    print(f"{tag}: {len(eps)} episodes")


ALL_B = ["random", "stalest", "flooded_first", "forecast", "greedy_unc", "oracle"]
OPS_B = ["random", "stalest", "flooded_first", "forecast", "greedy_unc"]


def main(suite):
    S = suite
    if S in ("core", "all"):
        run_suite("core", ALL_B, ["recurrent", "feedforward"])
    if S in ("ablation", "all"):
        run_suite("ablation", [], ["recurrent_noforecast", "recurrent_unc", "recurrent_iou"])
    if S in ("forecast", "all"):
        for mode in ["accurate", "noisy", "biased_low", "biased_high", "displaced", "uninformative", "missing"]:
            run_suite(f"forecast_{mode}", OPS_B, ["recurrent", "feedforward", "recurrent_noforecast"],
                      spec_kw={"forecast_mode": mode})
        run_suite("forecast_nohydro", OPS_B, ["recurrent", "feedforward", "recurrent_noforecast"],
                  env_kw={"hydro_obs": False})
    if S in ("sensor", "all"):
        for sensor in ["perfect", "degraded"]:
            run_suite(f"sensor_{sensor}", ALL_B, ["recurrent", "feedforward"], env_kw={"sensor": sensor})
    if S in ("budget", "all"):
        for k in [8, 32]:
            run_suite(f"budget_{k}", ALL_B, ["recurrent", "feedforward"], env_kw={"overrides": {"budget_per_pass": k}})
    if S in ("revisit", "all"):
        for h in [6, 24]:
            run_suite(f"revisit_{h}h", ALL_B, ["recurrent", "feedforward"], env_kw={"overrides": {"pass_interval_h": h}})
    if S in ("latency", "all"):
        run_suite("latency_1", ALL_B, ["recurrent", "feedforward"], env_kw={"overrides": {"latency_passes": 1}})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="core")
    main(ap.parse_args().suite)
