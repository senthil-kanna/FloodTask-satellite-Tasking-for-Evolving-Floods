"""Evaluate trained U-Net checkpoints on held-out TEST events (run once, after model selection is frozen).

Reports pooled and per-event IoU / precision / recall / F1 / boundary F1 / ECE, plus the Sen1Floods11
S1 Otsu-threshold labels as a classical reference. Usage: python -m floodtask.seg.evaluate --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from floodtask.eval.metrics import boundary_f1, confusion, expected_calibration_error, scores_from_confusion
from floodtask.seg.dataset import chips_for, normalize, read_chip
from floodtask.seg.unet import UNet
from floodtask.utils import DATA, load_json, provenance, RESULTS, save_json


def load_model(seed: int, device: str) -> UNet:
    m = UNet().to(device)
    m.load_state_dict(torch.load(RESULTS / "segmentation" / f"seed{seed}" / "best.pt", weights_only=True, map_location=torch.device(device)))
    return m.eval()


@torch.no_grad()
def predict(model, s1, stats, device):
    x, valid = normalize(s1, stats)
    p = model.predict_proba(torch.from_numpy(x)[None].to(device))[0].cpu().numpy()
    return p, valid


def evaluate_split(split: str, seeds: list[int]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stats = load_json(DATA / "processed" / "s1_norm_stats.json")
    chips = chips_for(split)
    data = {c["chip"]: read_chip(c["chip"]) for c in chips}
    res = {"split": split, "seeds": {}, "otsu": {}}
    for seed in seeds + ["otsu"]:
        model = None if seed == "otsu" else load_model(seed, device)
        per_event, per_chip, probs, truths = {}, [], [], []
        for c in chips:
            s1, lab, otsu = data[c["chip"]]
            v = lab >= 0
            if seed == "otsu":
                pred = otsu == 1
                v &= otsu >= 0
            else:
                p, valid = predict(model, s1, stats, device)
                v &= valid
                pred = p >= 0.5
                sel = np.flatnonzero(v.ravel())[:: 50]
                probs.append(p.ravel()[sel]); truths.append(lab.ravel()[sel])
            cm = np.array(confusion(pred, lab == 1, v))
            per_event.setdefault(c["event"], np.zeros(4, int))
            per_event[c["event"]] += cm
            per_chip.append({"chip": c["chip"], "event": c["event"], **scores_from_confusion(*cm),
                             "boundary_f1": boundary_f1(pred, lab == 1, tol=2, valid=v)})
        pooled = scores_from_confusion(*sum(per_event.values()))
        out = {"pooled": pooled,
               "per_event": {e: scores_from_confusion(*cm) for e, cm in per_event.items()},
               "mean_chip_iou": float(np.mean([r["iou"] for r in per_chip if r["tp"] + r["fn"] > 0])),
               "boundary_f1_mean": float(np.mean([r["boundary_f1"] for r in per_chip])),
               "per_chip": per_chip}
        if seed != "otsu":
            out["ece"] = expected_calibration_error(np.concatenate(probs), np.concatenate(truths))
            res["seeds"][str(seed)] = out
        else:
            res["otsu"] = out
        print(f"{split} {seed}: IoU {pooled['iou']:.4f} P {pooled['precision']:.3f} R {pooled['recall']:.3f} "
              + " ".join(f"{e}:{s['iou']:.3f}" for e, s in out["per_event"].items()))
    ious = [s["pooled"]["iou"] for s in res["seeds"].values()]
    res["iou_mean_over_seeds"], res["iou_std_over_seeds"] = float(np.mean(ious)), float(np.std(ious, ddof=1)) if len(ious) > 1 else 0.0
    res["provenance"] = provenance()
    save_json(res, RESULTS / "segmentation" / f"eval_{split}.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    evaluate_split(a.split, a.seeds)
