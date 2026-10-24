"""Tune heuristic baseline weights on VALIDATION-event scenes (never test). Criterion: mean per-pass IoU,
identical to the PPO checkpoint-selection criterion. Output: results/baseline_params.json"""
from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from floodtask.env.factory import make_env, specs
from floodtask.eval.runner import run_episode, summarize
from floodtask.policies.baselines import FloodedFirstPolicy, ForecastPolicy, GreedyUncertaintyPolicy
from floodtask.utils import provenance, RESULTS, save_json

N_EP = 48
GRIDS = {
    "greedy_unc": (GreedyUncertaintyPolicy, dict(tau_h=[12, 24, 48, 96], w_terrain=[0.5, 1, 2], w_flood=[0.5, 1, 2],
                                                 w_rain=[0, 0.5, 2], w0=[0.02, 0.1])),
    "forecast": (ForecastPolicy, dict(w_susc=[0.1, 0.5, 2], w_age=[0, 0.2, 1])),
    "flooded_first": (FloodedFirstPolicy, dict(w_terrain=[0.1, 0.3, 1], w_age=[0, 0.1, 0.5])),
}


def score(args):
    name, kw = args
    cls = GRIDS[name][0]
    env = make_env("val")
    pol = cls(**kw)
    vals = [summarize(run_episode(env, pol, s))["mean_iou"] for s in specs("val", N_EP)]
    return name, kw, float(np.nanmean(vals))


def main():
    jobs = [(n, dict(zip(g.keys(), combo))) for n, (_, g) in GRIDS.items() for combo in itertools.product(*g.values())]
    print(len(jobs), "configurations")
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(score, jobs))
    out = {}
    for name in GRIDS:
        rows = sorted([r for r in res if r[0] == name], key=lambda r: -r[2])
        out[name] = {"params": rows[0][1], "val_mean_iou": rows[0][2],
                     "all": [{"params": r[1], "val_mean_iou": r[2]} for r in rows]}
        print(name, rows[0][1], round(rows[0][2], 4), "(worst", round(rows[-1][2], 4), ")")
    out["n_val_episodes"] = N_EP
    out["provenance"] = provenance()
    save_json(out, RESULTS / "baseline_params.json")


if __name__ == "__main__":
    main()
