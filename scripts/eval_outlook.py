"""Validate the flood outlook (floodtask/forecast/outlook.py) against the hidden simulated flood state.

  python scripts/eval_outlook.py --split val     # choose the weight temperature (TAU) on validation scenes
  python scripts/eval_outlook.py --split test    # report on held-out test scenes with the chosen TAU
The tasking policy that builds the map is the tuned greedy-uncertainty baseline (the outlook does not depend on which
policy chose the tiles beyond the map it is given). Outlooks are issued after passes 3, 6 and 9 and scored at +4, +6, +12 and +24 h
and later horizons against the true flood extent, next to two references: persistence (today's map stays) and the
unweighted ensemble (tau = 0). Writes results/outlook_<split>.json.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from floodtask.env.factory import make_env, scene_names, specs
from floodtask.eval.metrics import scores_from_confusion
from floodtask.forecast.outlook import TAU, flood_outlook
from floodtask.hydro.simulator import flood_mask
from floodtask.utils import provenance, RESULTS, save_json

ISSUE_PASSES = (2, 5, 8)          # 0-based: after the 3rd, 6th and 9th pass
HORIZONS = (4, 6, 12, 24)
TAUS_VAL = (0.0, 0.005, 0.02, 0.1)


def score(prob, truth):
    pred = prob >= 0.5
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    m = scores_from_confusion(tp, fp, fn, truth.size - tp - fp - fn)
    return {"iou": m["iou"], "brier": float(np.mean((prob - truth) ** 2)), "area_pred": float(pred.mean()),
            "area_true": float(truth.mean())}


def episode(args):
    split, spec, taus = args
    from run_experiments import tuned
    env = make_env(split)
    pol = tuned("greedy_unc")
    obs = env.reset(spec)
    pol.reset(0)
    rows, done = [], False
    while not done:
        a = pol.act(obs, None)
        obs, _, done, info = env.step(a)
        if "pass_metrics" in info and info["pass_metrics"]["pass"] in ISSUE_PASSES:
            p = info["pass_metrics"]["pass"]
            t = int(env.pass_t[p])
            truth = {h: flood_mask(env.sc, env._level[t + h]) for h in HORIZONS}
            now_truth = flood_mask(env.sc, env._level[t])
            base = {"scene": spec.scene, "seed": spec.seed, "pass": p}
            for h in HORIZONS:
                rows.append({**base, "h": h, "method": "persistence", **score(env.est_prob.astype(np.float32), truth[h]),
                             "changed_frac": float((truth[h] != now_truth).mean())})
            for tau in taus:
                o = flood_outlook(env, p, HORIZONS, tau=tau, seed=spec.seed)
                for h in HORIZONS:
                    rows.append({**base, "h": h, "method": f"outlook_tau{tau}", "ess": o["ess"], **score(o[h], truth[h])})
    return rows


def main(split, n):
    import sys
    sys.path.insert(0, str(RESULTS.parent / "scripts"))
    taus = TAUS_VAL if split == "val" else (0.0, TAU)
    sp = specs(split, n)
    with ProcessPoolExecutor(max_workers=8, initializer=sys.path.insert, initargs=(0, str(RESULTS.parent / "scripts"))) as ex:
        rows = [r for rs in ex.map(episode, [(split, s, taus) for s in sp]) for r in rs]
    import pandas as pd
    df = pd.DataFrame(rows)
    flood = df[df.area_true >= 0.001]
    summ = flood.groupby(["h", "method"])[["iou", "brier", "area_pred", "area_true"]].mean().round(4)
    print(summ.to_string())
    # paired difference outlook (chosen tau) - persistence on IoU, per scene
    pair = {}
    for h in HORIZONS:
        a = flood[(flood.h == h) & (flood.method == f"outlook_tau{TAU}")].set_index(["scene", "seed", "pass"])
        b = flood[(flood.h == h) & (flood.method == "persistence")].set_index(["scene", "seed", "pass"])
        d = (a["iou"] - b["iou"]).groupby("scene").mean()
        rng = np.random.default_rng(0)
        boot = [rng.choice(d.values, d.size).mean() for _ in range(5000)]
        pair[h] = {"iou_diff_vs_persistence": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                   "n_scenes": int(d.size), "brier_outlook": float(a["brier"].mean()), "brier_persistence": float(b["brier"].mean())}
    print(pair)
    save_json({"split": split, "n_episodes": n, "issue_passes": ISSUE_PASSES, "tau_chosen": TAU,
               "summary_flood_present": summ.reset_index().to_dict("records"), "paired_vs_persistence": pair,
               "rows": rows, "provenance": provenance()}, RESULTS / f"outlook_{split}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=None, help="episodes (default: one per scene)")
    a = ap.parse_args()
    main(a.split, a.n or len(scene_names(a.split)))
