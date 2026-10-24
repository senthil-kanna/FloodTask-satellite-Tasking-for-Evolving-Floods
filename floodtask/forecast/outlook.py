"""Flood outlook: probability that each cell is flooded some hours after a pass, from decision-time information only.

Ensemble of the conceptual reach-storage model (floodtask/hydro/simulator.py):
  - members sample physical parameters from the training prior (configs/env.yaml `physics`, after calibration);
  - rain input: past hours = rain observations (true field x lognormal error per day and grid cell, sd 0.2, the same
    error level as the agent's rain_obs channel); future hours = the 24 h and 72 h accumulation forecast issued at the
    pass (env._forecast, which already carries its error model), spread evenly over each window. When the forecast is
    missing, future rain is the climatological mean rate;
  - members are weighted by how well their simulated flood extent at the pass agrees with the current fused map on
    validly observed cells (likelihood weighting, as in a particle filter);
  - outlook = weighted share of members that flood each cell.
The truth fields of the environment (flood state, true parameters) are never read. The past rain field is read only
through the observation error above. Inherits every simulator omission (docs/limitations.md). In the Assam replays the
"forecast" is an ERA5 pseudo-forecast built after the event.
"""
from __future__ import annotations

import numpy as np

from floodtask.hydro.simulator import EXT, NT, SimParams, flood_mask, simulate, upstream_mask

N_MEMBERS = 48
TAU = 0.02          # weight temperature on the mismatch fraction; chosen on validation scenes (scripts/eval_outlook.py)
RAIN_OBS_SD = 0.2


def _sample_params(rng, ph, Z):
    u = lambda k: rng.uniform(*ph[k])
    return SimParams(runoff_coef=u("runoff_coef"), residence_h=rng.uniform(*ph["residence_h"], Z),
                     ext_scale=float(np.exp(rng.uniform(*np.log(ph["ext_scale"])))), ext_lag_h=u("ext_lag_h"),
                     loss_mm=u("loss_mm"))


def _rain_input(env, p_idx: int, horizon_h: int, rng) -> np.ndarray:
    """Hourly rain on the extended grid up to pass time + horizon: observed past, forecast future."""
    t = int(env.pass_t[p_idx])
    hours = t + horizon_h + 1
    past = env._rain[:t].astype(np.float64)
    days = -(-max(t, 1) // 24)
    err = np.exp(rng.normal(0, RAIN_OBS_SD, (days,) + past.shape[1:]))
    past = past * np.repeat(err, 24, axis=0)[:t]
    fut = np.zeros((hours - t,) + env._rain.shape[1:])
    f = env._forecast(p_idx)
    NE = env._rain.shape[1]
    up = upstream_mask(env.sc["upstream_side"])
    if f["missing"]:
        clim = np.mean(env.cfg["physics"]["storm_total_mm"]) / (env.P * env.dt)
        fut[:] = clim
    else:
        w24, u24 = f[24]
        w72, u72 = f[72]
        rate = lambda acc, h: np.maximum(acc, 0) / h
        for lo, hi, win, upv in ((0, 24, rate(w24, 24), rate(u24, 24)), (24, 72, rate(w72 - w24, 48), rate(u72 - u24, 48))):
            hi = min(hi, hours - t)
            if hi <= lo:
                continue
            field = np.full((NE, NE), float(np.mean(win)))
            field[up] = upv
            field[EXT:EXT + NT, EXT:EXT + NT] = np.asarray(win).reshape(NT, NT)
            fut[lo:hi] = field
        if hours - t > 72:      # beyond the forecast range: climatology
            fut[72:] = np.mean(env.cfg["physics"]["storm_total_mm"]) / (env.P * env.dt)
    return np.concatenate([past, fut]).astype(np.float32)


def flood_outlook(env, p_idx: int, horizons=(24, 72), n_members: int = N_MEMBERS, tau: float = TAU, seed: int = 0) -> dict:
    """Outlook issued at pass p_idx, after that pass's acquisitions are in the map (call it once the pass has ended).
    Returns {h: P(flooded) [SCENE,SCENE] float32}, plus 'now' (weighted ensemble extent at the pass), 'weights' and
    'ess' (effective ensemble size)."""
    t = int(env.pass_t[p_idx])
    hmax = max(horizons)
    rng = np.random.default_rng(seed + 7919 * p_idx)
    rain = _rain_input(env, p_idx, hmax, rng)
    sc = env.sc
    valid = env.est_valid
    obs = env.est_prob >= 0.5
    members, mism = [], []
    for _ in range(n_members):
        par = _sample_params(rng, env.cfg["physics"], sc["Z"])
        lvl = simulate(sc, rain, par)
        now = flood_mask(sc, lvl[t])
        members.append((now, {h: flood_mask(sc, lvl[min(t + h, rain.shape[0] - 1)]) for h in horizons}))
        mism.append(float((now[valid] != obs[valid]).mean()) if valid.any() else 0.0)
    mism = np.array(mism)
    w = np.exp(-(mism - mism.min()) / tau) if tau > 0 else np.ones(n_members)
    w /= w.sum()
    out = {h: sum(wi * m[1][h] for wi, m in zip(w, members)).astype(np.float32) for h in horizons}
    out["now"] = sum(wi * m[0] for wi, m in zip(w, members)).astype(np.float32)
    out["weights"], out["ess"] = w, float(1 / np.sum(w ** 2))
    return out


def persistence(env) -> np.ndarray:
    """Reference forecast: the current fused map stays as it is."""
    return env.est_prob.astype(np.float32)
