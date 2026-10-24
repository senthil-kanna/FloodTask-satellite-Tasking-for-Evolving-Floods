"""Calibrate and validate the conceptual flood model against real data.

Validation (all events, reported per event; no fitting):
  HAND-AUC: does low HAND predict hand-labelled water in Sen1Floods11 chips on the 60 m scene grid?
Calibration (TRAINING events only):
  storm_total_mm range from ERA5 10-day pre-acquisition totals of training events (widened for ERA5's
  smoothing of local extremes), then a single scale factor on the external-inflow range chosen so that the
  simulated flooded-fraction distribution at mid-episode matches the hand-labelled water fraction of the
  training scenes (two-sample KS distance).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps

from floodtask.env.factory import env_config, sensor_params
from floodtask.env.tasking_env import EpisodeSpec, FloodTaskingEnv
from floodtask.hydro.simulator import load_scene
from floodtask.utils import DATA, load_json, load_yaml, provenance, RESULTS, ROOT, save_json


def hand_auc(scenes):
    out = {}
    for s in scenes:
        sc = load_scene(s["scene"])
        ok = sc["label_valid"] >= 0.9
        if ok.sum() < 200:
            continue
        y = sc["label_water"][ok] >= 0.5
        if y.sum() < 20 or (~y).sum() < 20:
            continue
        auc = sps.mannwhitneyu(-sc["hand"][ok][y], -sc["hand"][ok][~y]).statistic / (y.sum() * (~y).sum())
        out.setdefault(s["event"], []).append({"scene": s["scene"], "auc": float(auc), "n_cells": int(ok.sum()),
                                               "water_frac": float(y.mean())})
    return out


def observed_fraction(scenes):
    vals = []
    for s in scenes:
        sc = load_scene(s["scene"])
        ok = sc["label_valid"] >= 0.9
        if ok.sum() >= 200:
            vals.append(float((sc["label_water"][ok] >= 0.5).mean()))
    return np.array(vals)


def simulated_fraction(cfg, scenes, n=120):
    env = FloodTaskingEnv(cfg, sensor_params(), [s["scene"] for s in scenes], seed=0)
    vals = []
    for i in range(n):
        env.reset(EpisodeSpec(scenes[i % len(scenes)]["scene"], 500_000 + i))
        sc = env.sc
        ok = sc["label_valid"] >= 0.9
        mid = env._truth_eval[env.P // 2]
        vals.append(float(mid[ok].mean()) if ok.sum() >= 200 else float(mid.mean()))
    return np.array(vals)


def main():
    scenes = load_json(DATA / "processed" / "scenes.json")
    tr_events = load_yaml(ROOT / "configs" / "data.yaml")["split"]["train"]
    train = [s for s in scenes if s["event"] in tr_events]
    auc = hand_auc(scenes)
    for e, rows in sorted(auc.items()):
        print(f"HAND-AUC {e:10s} median {np.median([r['auc'] for r in rows]):.3f} over {len(rows)} scenes")
    era5 = load_json(DATA / "processed" / "era5_event_rainfall.json")
    totals = np.array([era5[e]["total_10d_before_mm"] for e in tr_events])
    storm_range = [float(max(20.0, totals.min())), float(max(totals.max() * 1.5, 100.0))]
    print("ERA5 training-event 10-day totals:", dict(zip(tr_events, totals.round(1))), "->", storm_range)
    obs = observed_fraction(train)
    prior = load_yaml(ROOT / "configs" / "env.yaml")["physics"]           # uncalibrated prior ranges
    (lo, hi), (rlo, rhi) = prior["ext_scale"], prior["runoff_coef"]
    best = None
    grid = [(mr, me, L) for mr in [0.125, 0.25, 0.5, 1.0] for me in [0.25, 1.0, 4.0] for L in [0, 5, 10, 20, 40]]
    for mr, me, L in grid:   # runoff scale x external inflow scale x overbank threshold (mm over zone area)
            phys = {"storm_total_mm": storm_range, "ext_scale": [lo * me, hi * me], "runoff_coef": [rlo * mr, rhi * mr],
                    "loss_mm": [0.5 * L, 1.5 * L]}
            sim = simulated_fraction(env_config({"physics": phys}), train)
            ks = sps.ks_2samp(sim, obs).statistic
            print(f"runoff x{mr} ext x{me} loss {L}mm: sim median {np.median(sim):.4f} vs observed {np.median(obs):.4f}  KS {ks:.3f}", flush=True)
            if best is None or ks < best[0]:
                best = (ks, phys, sim, mr, me, L)
    ks, phys, sim, m, me, L = best
    save_json({"physics": phys, "runoff_multiplier": m, "ext_scale_multiplier": me, "loss_mm_centre": L, "ks_statistic": ks,
               "observed_frac_train": obs.tolist(), "simulated_frac_train": sim.tolist(), "hand_auc": auc,
               "era5_train_totals": dict(zip(tr_events, totals)), "provenance": provenance()},
              RESULTS / "sim_calibration.json")
    print(f"chosen runoff x{m}, ext x{me}, loss {L} mm (KS {ks:.3f})")



if __name__ == "__main__":
    main()
