"""Binary flood-map metrics. All functions ignore pixels where `valid` is False.

Conventions (fixed before test evaluation):
- IoU = TP / (TP + FP + FN); defined as 1.0 when TP+FP+FN == 0 (both maps empty).
- Precision / recall / F1 defined as 1.0 when their denominator is zero and no errors exist, else 0.
- Area bias = (predicted flooded area - true flooded area) / max(true area, 1 pixel), signed.
- Boundary F1: boundaries are 4-connected edge pixels; match tolerance `tol` pixels (Chebyshev).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def confusion(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None) -> tuple[int, int, int, int]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    if valid is None:
        valid = np.ones_like(pred, bool)
    tp = int(np.sum(pred & truth & valid))
    fp = int(np.sum(pred & ~truth & valid))
    fn = int(np.sum(~pred & truth & valid))
    tn = int(np.sum(~pred & ~truth & valid))
    return tp, fp, fn, tn


def scores_from_confusion(tp, fp, fn, tn) -> dict:
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    prec = (1.0 if fn == 0 else 0.0) if tp + fp == 0 else tp / (tp + fp)
    rec = (1.0 if fp == 0 else 0.0) if tp + fn == 0 else tp / (tp + fn)
    f1 = 1.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    n = tp + fp + fn + tn
    return {
        "iou": iou, "precision": prec, "recall": rec, "f1": f1,
        "pixel_error": (fp + fn) / max(n, 1),
        "area_bias": ((tp + fp) - (tp + fn)) / max(tp + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def map_scores(pred, truth, valid=None) -> dict:
    return scores_from_confusion(*confusion(pred, truth, valid))


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), border_value=1)


def boundary_f1(pred: np.ndarray, truth: np.ndarray, tol: int = 2, valid: np.ndarray | None = None) -> float:
    pb, tb = _boundary(pred.astype(bool)), _boundary(truth.astype(bool))
    if valid is not None:
        pb, tb = pb & valid, tb & valid
    if pb.sum() == 0 and tb.sum() == 0:
        return 1.0
    if pb.sum() == 0 or tb.sum() == 0:
        return 0.0
    st = np.ones((2 * tol + 1, 2 * tol + 1), bool)
    p_hit = np.sum(pb & ndimage.binary_dilation(tb, st)) / pb.sum()
    r_hit = np.sum(tb & ndimage.binary_dilation(pb, st)) / tb.sum()
    return 0.0 if p_hit + r_hit == 0 else 2 * p_hit * r_hit / (p_hit + r_hit)


def expected_calibration_error(prob: np.ndarray, truth: np.ndarray, n_bins: int = 15) -> float:
    prob, truth = prob.ravel(), truth.ravel().astype(float)
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob >= 0.5) == (truth > 0.5)).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
