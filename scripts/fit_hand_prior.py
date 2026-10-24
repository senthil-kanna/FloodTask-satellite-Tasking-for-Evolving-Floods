"""Fit the terrain prior P(water | HAND) used by the shared map estimator, on TRAINING-event labels only.

Logistic regression of hand-labelled water (60 m cells, label coverage >= 90%) on log1p(HAND).
Also stores pi_train, the water fraction of the U-Net training pixels (seed 0), needed for the prior-shift
correction: logit p_post = logit p_obs - logit pi_train + logit P(water | HAND).
Output: results/hand_prior.json
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from floodtask.hydro.simulator import load_scene
from floodtask.utils import DATA, load_json, load_yaml, provenance, RESULTS, ROOT, save_json


def main():
    train = set(load_yaml(ROOT / "configs" / "data.yaml")["split"]["train"])
    xs, ys = [], []
    for s in load_json(DATA / "processed" / "scenes.json"):
        if s["event"] not in train:
            continue
        sc = load_scene(s["scene"])
        ok = sc["label_valid"] >= 0.9
        xs.append(np.log1p(sc["hand"][ok])); ys.append(sc["label_water"][ok] >= 0.5)
    x, y = np.concatenate(xs), np.concatenate(ys).astype(float)

    def nll(w):
        z = w[0] + w[1] * x
        return np.mean(np.logaddexp(0, z) - y * z)

    w = minimize(nll, np.zeros(2), method="BFGS").x
    pi = load_json(RESULTS / "segmentation" / "seed0" / "train_log.json")["train_water_fraction"]
    p = 1 / (1 + np.exp(-(w[0] + w[1] * x)))
    out = {"intercept": float(w[0]), "slope_log1p_hand": float(w[1]), "pi_train": float(pi), "n_cells": int(x.size),
           "train_water_rate": float(y.mean()),
           "prior_at_hand_m": {h: float(1 / (1 + np.exp(-(w[0] + w[1] * np.log1p(h))))) for h in (0, 1, 2, 5, 10, 20)},
           "mean_prior_on_water_cells": float(p[y == 1].mean()), "mean_prior_on_dry_cells": float(p[y == 0].mean()),
           "provenance": provenance()}
    save_json(out, RESULTS / "hand_prior.json")
    print({k: v for k, v in out.items() if k != "provenance"})


if __name__ == "__main__":
    main()
