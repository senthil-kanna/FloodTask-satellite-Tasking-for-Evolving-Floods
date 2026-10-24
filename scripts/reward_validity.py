"""R12 reward-validity check on VALIDATION episodes: does episode return rank episodes/policies like the
independent primary endpoint (mean IoU over flood-present passes)? Output: results/reward_validity_val.json"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import spearmanr

from floodtask.env.factory import make_env, specs
from floodtask.eval.runner import run_episode, summarize
from floodtask.utils import provenance, RESULTS, save_json

sys.path.insert(0, str(RESULTS.parent / "scripts"))
from run_experiments import tuned  # noqa: E402

POLICIES = ["random", "stalest", "flooded_first", "forecast", "greedy_unc", "oracle"]


def main(n=32):
    out = {}
    for kind, fnw in (("err", 1.0), ("err", 3.0), ("err", 10.0), ("err", 30.0), ("iou_diff", 1.0)):
        env = make_env("val", overrides={"reward": {"kind": kind, "unc_coef": 0.0, "fn_weight": fnw}})
        kind = f"{kind}_fnw{fnw:g}" if kind == "err" else kind
        rows = [{**summarize(run_episode(env, tuned(p), s)), "policy": p} for p in POLICIES for s in specs("val", n)]
        ok = [r for r in rows if np.isfinite(r["mean_iou"])]
        pm = {p: (float(np.mean([r["return"] for r in ok if r["policy"] == p])),
                  float(np.mean([r["mean_iou"] for r in ok if r["policy"] == p]))) for p in POLICIES}
        out[kind] = {"spearman_episode_level": float(spearmanr([r["return"] for r in ok], [r["mean_iou"] for r in ok]).correlation),
                     "spearman_policy_level": float(spearmanr([a for a, _ in pm.values()], [b for _, b in pm.values()]).correlation),
                     "policy_means_return_iou": pm, "n_episodes_used": len(ok), "n_episodes_no_flood": len(rows) - len(ok)}
        print(kind, {k: v for k, v in out[kind].items() if k != "policy_means_return_iou"})
        print("   ", {p: (round(a, 2), round(b, 4)) for p, (a, b) in pm.items()})
    save_json({**out, "split": "val", "provenance": provenance()}, RESULTS / "reward_validity_val.json")


if __name__ == "__main__":
    main()
