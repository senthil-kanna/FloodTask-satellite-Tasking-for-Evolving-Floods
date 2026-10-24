"""Empirical segmentation-error model linking the trained U-Net to the tasking environment.

An acquired tile does NOT disclose ground truth. It yields a water-probability map
    p = sigmoid( a * (2 * G_b(truth) - 1) + bias + sigma * N_corr )
where G_b is a Gaussian blur (boundary uncertainty) and N_corr is unit-variance spatially correlated noise.
Parameters are fitted (scripts/fit_sensor_model.py) so that this model reproduces the U-Net's error statistics
on VALIDATION-event chips aggregated to the 60 m simulation grid. Test events are used only to check the fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SensorParams:
    a: float = 3.0
    bias: float = -0.3
    sigma: float = 1.2
    corr: float = 1.5
    blur: float = 0.7
    nodata_tile_prob: float = 0.05

    def to_dict(self):
        return asdict(self)


PERFECT = SensorParams(a=12.0, bias=0.0, sigma=0.0, corr=1.0, blur=0.0, nodata_tile_prob=0.0)


def observe(truth_window: np.ndarray, pad: int, sp: SensorParams, rng: np.random.Generator):
    """truth_window: bool array of the tile with `pad` cells of context on each side (context is clipped at
    scene borders by the caller). Returns (prob, valid) for the central tile."""
    t = truth_window.astype(np.float32)
    soft = gaussian_filter(t, sp.blur, mode="nearest") if sp.blur > 0 else t
    logit = sp.a * (2 * soft - 1) + sp.bias
    if sp.sigma > 0:
        n = gaussian_filter(rng.standard_normal(t.shape).astype(np.float32), sp.corr, mode="wrap")
        n /= max(n.std(), 1e-6)
        logit = logit + sp.sigma * n
    prob = 1 / (1 + np.exp(-logit))
    h, w = t.shape
    prob = prob[pad:h - pad, pad:w - pad]
    valid = np.ones_like(prob, bool)
    if rng.random() < sp.nodata_tile_prob:  # partial nodata block (border noise, layover, processing gaps)
        r0, c0 = rng.integers(0, prob.shape[0], 2)
        valid[r0:, c0:] = False
    return prob, valid
