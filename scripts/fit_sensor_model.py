"""Fit the environment's segmentation-error model to the trained U-Net.

1. Run the U-Net (seed 0, temperature-calibrated) on VALIDATION-event chips.
2. Aggregate label and probability to the 60 m simulation grid (6x6 blocks of 10 m pixels).
3. Grid-search SensorParams so that applying the sensor model to the 60 m LABEL masks reproduces
   the U-Net's 60 m error statistics (FPR, FNR, IoU, mean entropy, ECE).
4. Report the same statistics on TEST events as an out-of-sample check (not used for fitting).
Also estimates the probability that a 720 m tile contains a nodata block, from TRAINING chips.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from floodtask.env.sensor import SensorParams, observe
from floodtask.env.tasking_env import binary_entropy
from floodtask.eval.metrics import expected_calibration_error
from floodtask.seg.dataset import chips_for, read_chip
from floodtask.seg.evaluate import load_model, predict
from floodtask.utils import DATA, load_json, provenance, RESULTS, save_json

B = 6  # 10 m pixels per 60 m cell


def block(a):
    n = (a.shape[0] // B) * B
    return a[:n, :n].reshape(n // B, B, n // B, B).mean((1, 3))


def chip_60m(model, c, stats, device):
    s1, lab, _ = read_chip(c["chip"])
    p, valid = predict(model, s1, stats, device)
    v = (lab >= 0) & valid
    vf = block(v.astype(float))
    wf = block(((lab == 1) & v).astype(float)) / np.maximum(vf, 1e-9)
    pf = block(np.where(v, p, 0)) / np.maximum(vf, 1e-9)
    ok = vf >= 0.9
    return wf >= 0.5, pf, ok


def stats_of(truth_list, prob_list, ok_list):
    t = np.concatenate([x[o] for x, o in zip(truth_list, ok_list)])
    p = np.concatenate([x[o] for x, o in zip(prob_list, ok_list)])
    pred = p >= 0.5
    tp, fp, fn = (pred & t).sum(), (pred & ~t).sum(), (~pred & t).sum()
    return {"fpr": fp / max((~t).sum(), 1), "fnr": fn / max(t.sum(), 1), "iou": tp / max(tp + fp + fn, 1),
            "entropy": float(binary_entropy(p).mean()), "ece": expected_calibration_error(p, t)}


def simulate(truths, oks, sp, seed=0):
    rng = np.random.default_rng(seed)
    probs = []
    for t in truths:
        pad = 3
        prob, _ = observe(np.pad(t, pad, mode="edge"), pad, SensorParams(**{**sp.to_dict(), "nodata_tile_prob": 0}), rng)
        probs.append(prob)
    return stats_of(truths, probs, oks)


def loss(a, b):
    return sum(((a[k] - b[k]) / max(abs(b[k]), 0.02)) ** 2 for k in ("fpr", "fnr", "iou", "entropy", "ece"))


def nodata_tile_prob():
    hits = n = 0
    for c in chips_for("train"):
        s1, _, _ = read_chip(c["chip"])
        bad = ~np.isfinite(s1).all(0)
        for r in range(0, 512 - 72 + 1, 72):
            for cc in range(0, 512 - 72 + 1, 72):
                n += 1
                hits += bad[r:r + 72, cc:cc + 72].mean() > 0.05
    return hits / n


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_json(DATA / "processed" / "s1_norm_stats.json")
    model = load_model(0, device)
    sets = {}
    for split in ("val", "test"):
        rows = [chip_60m(model, c, stats, device) for c in chips_for(split)]
        sets[split] = [list(x) for x in zip(*rows)]
    target = stats_of(*sets["val"])
    print("U-Net 60 m val stats:", {k: round(float(v), 4) for k, v in target.items()})
    grid = itertools.product([1.5, 2, 3, 4, 6, 8], [-2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.3], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [0.5, 1.0, 2.0], [0.0, 0.5, 1.0, 1.5])
    best = None
    for a, b, s, c, bl in grid:
        sp = SensorParams(a=a, bias=b, sigma=s, corr=c, blur=bl)
        st = simulate(sets["val"][0], sets["val"][2], sp)
        l = loss(st, target)
        if best is None or l < best[0]:
            best = (l, sp, st)
    l, sp, st = best
    sp.nodata_tile_prob = float(nodata_tile_prob())
    test_real = stats_of(*sets["test"])
    test_sim = simulate(sets["test"][0], sets["test"][2], sp, seed=1)
    print("fitted:", sp.to_dict(), "loss", round(l, 4))
    print("val sim  :", {k: round(float(v), 4) for k, v in st.items()})
    print("test real:", {k: round(float(v), 4) for k, v in test_real.items()})
    print("test sim :", {k: round(float(v), 4) for k, v in test_sim.items()})
    save_json({"params": sp.to_dict(), "fit_loss": l, "val_unet_60m": target, "val_sim_60m": st,
               "test_unet_60m_check": test_real, "test_sim_60m_check": test_sim, "provenance": provenance()},
              RESULTS / "sensor_model.json")


if __name__ == "__main__":
    main()
