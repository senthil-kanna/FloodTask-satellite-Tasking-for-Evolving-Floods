"""Real-event evaluation on Sentinel-1 RTC time series (default: Assam 2016 and 2022, India test scenes).

  python scripts/run_real_event.py --event assam2016
Policies are frozen (best-validation checkpoints / validation-tuned baselines). Two forecast conditions:
'operational' (ERA5 PSEUDO-forecast with the training error model) and 'missing'.
For events with a hand-label date, the reference masks and the deployed U-Net are also checked against the
Sen1Floods11 hand labels for that date (domain-shift and reference-quality check).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
import torch
from rasterio.warp import Resampling, reproject

from floodtask.data.real_event import event_cfg, ROOT_OUT, scenes
from floodtask.data.scenes import SCENE
from floodtask.env.factory import env_config, sensor_params, VARIANTS
from floodtask.env.replay_env import ReplayEnv
from floodtask.env.tasking_env import EpisodeSpec
from floodtask.eval.metrics import scores_from_confusion
from floodtask.eval.runner import RLPolicy, run_episode
from floodtask.rl.model import ActorCritic
from floodtask.utils import DATA, provenance, RESULTS, save_json

import sys
sys.path.insert(0, str((RESULTS.parent / "scripts")))
from run_experiments import OPS_B, RL_SEEDS, tuned  # noqa: E402

EPISODE_SEEDS = [0, 1, 2]   # vary only policy tie-breaking / sampling; the observations are real and fixed


def label_check(key: str) -> dict | None:
    """Compare reference masks and deployed-U-Net observations with hand labels on the label date (60 m grid)."""
    ev = event_cfg(key)
    if "label_date" not in ev:
        return None
    tot = {"reference": np.zeros(4, int), "unet_obs": np.zeros(4, int)}
    n_cells = 0
    for s in scenes(key):
        m = np.load(ROOT_OUT / key / f"{s['scene']}_masks.npz")
        dates = [str(d) for d in m["dates"]]
        if ev["label_date"] not in dates:
            continue
        i = dates.index(ev["label_date"])
        sc = np.load(DATA / "processed" / "scenes" / f"{s['scene']}.npz")
        ok = (sc["label_valid"] >= 0.9)
        truth = sc["label_water"] >= 0.5
        for name, pred, valid in (("reference", m["ref"][i], ok & m["ref_valid"][i]),
                                  ("unet_obs", m["obs_prob"][i] >= 0.5, ok & m["obs_valid"][i])):
            tp = int((pred & truth & valid).sum()); fp = int((pred & ~truth & valid).sum())
            fn = int((~pred & truth & valid).sum()); tn = int(valid.sum()) - tp - fp - fn
            tot[name] += [tp, fp, fn, tn]
        n_cells += int(ok.sum())
    return {"label_date": ev["label_date"], "n_labelled_cells_60m": n_cells,
            **{k: scores_from_confusion(*v) for k, v in tot.items()}}


def main(key: str):
    names = [s["scene"] for s in scenes(key) if (ROOT_OUT / key / f"{s['scene']}_masks.npz").exists()]
    cfg = env_config()
    eps = []
    for mode in ("operational", "missing"):
        env = ReplayEnv(cfg, sensor_params(), names, key)
        sp = [EpisodeSpec(n, s, mode) for n in names for s in EPISODE_SEEDS]
        for b in OPS_B:
            eps += [{**run_episode(env, tuned(b), x), "train_seed": -1} for x in sp]
        for v in ("recurrent", "feedforward", "recurrent_noforecast"):
            for seed in RL_SEEDS:
                path = RESULTS / "rl" / v / f"seed{seed}" / "best.pt"
                if not path.exists():
                    continue
                m = ActorCritic(recurrent=VARIANTS[v]["recurrent"])
                m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                envv = ReplayEnv(cfg, sensor_params(), names, key, use_forecast=VARIANTS[v]["use_forecast"])
                pol = RLPolicy(m, "cpu", name=f"ppo_{v}")
                eps += [{**run_episode(envv, pol, x), "train_seed": seed} for x in sp]
        print(key, mode, "done")
    out = RESULTS / "experiments" / f"real_{key}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(json.dumps(e) + "\n")
    save_json({"event": key, "scenes": names, "n_episodes": len(eps), "label_check": label_check(key),
               "forecast_note": "operational = ERA5 pseudo-forecast (retrospective), NOT archived NWP",
               "provenance": provenance()}, out / "meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="assam2016")
    main(ap.parse_args().event)
