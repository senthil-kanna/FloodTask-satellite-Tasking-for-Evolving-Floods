"""Select the HAND drainage threshold on TRAINING events only.

Criterion: median (over scenes with labels) of HAND-AUC = P(HAND of a labelled-water cell < HAND of a labelled-dry cell),
plus the fraction of HAND~0 cells that are labelled water (reported, not optimised). Uses a fixed subsample of
training scenes (every 3rd scene centre per event) for speed. Output: results/channel_threshold.json
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy import stats as sps

from floodtask.data.scenes import build_scene, scene_centres
from floodtask.utils import DATA, load_json, load_yaml, provenance, RESULTS, ROOT, save_json

CANDIDATES = [200, 1000, 3000, 10000, 30000]   # cells of 60 m: 0.72 to 108 km2


def main():
    train = load_yaml(ROOT / "configs" / "data.yaml")["split"]["train"]
    chips = load_json(DATA / "raw" / "sen1floods11" / "chips.json")
    res = {}
    tmp = Path(tempfile.mkdtemp())
    for ch in CANDIDATES:
        aucs, w0 = [], []
        for ev in train:
            ev_chips = [c for c in chips if c["event"] == ev]
            for i, (lon, lat) in enumerate(scene_centres(ev_chips)[::3]):
                m = build_scene(ev, i, lon, lat, ev_chips, out_dir=tmp, channel=ch)
                if not m:
                    continue
                d = np.load(tmp / f"{m['scene']}.npz")
                ok = d["label_valid"] >= 0.9
                y = d["label_water"][ok] >= 0.5
                h = d["hand"][ok]
                if y.sum() < 20 or (~y).sum() < 20:
                    continue
                aucs.append(sps.mannwhitneyu(-h[y], -h[~y]).statistic / (y.sum() * (~y).sum()))
                if (h < 0.05).sum() > 20:
                    w0.append(y[h < 0.05].mean())
        res[ch] = {"median_auc": float(np.median(aucs)), "n_scenes": len(aucs),
                   "water_frac_among_hand0": float(np.median(w0)) if w0 else None}
        print(ch, res[ch], flush=True)
    best = max(res, key=lambda k: res[k]["median_auc"])
    save_json({"channel_cells": best, "cell_m": 60, "candidates": res, "events": train, "provenance": provenance()},
              RESULTS / "channel_threshold.json")
    print("selected", best)


if __name__ == "__main__":
    main()
