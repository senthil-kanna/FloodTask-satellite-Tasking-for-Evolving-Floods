"""Real-event replay environment (Sentinel-1 RTC time series; default Assam 2016 / 2022).

Differences from the simulator environment:
  * passes  = actual S1 acquisition dates over the scene (irregular intervals);
  * feasible tiles = tiles with >= 50% valid S1 data on that date (real swath coverage), so
    budget-limited tasking is an emulation: Sentinel-1 is not tasked per tile in reality;
  * an acquired tile delivers the deployed U-Net's (seed 0) real prediction for that date;
  * truth = reference mask (U-Net seeds 1+2 ensemble AND VH-Otsu agreement); evaluation uses only
    reference-valid cells (cells where the two methods disagree are excluded);
  * rainfall: ERA5 daily totals (RETROSPECTIVE reanalysis) spread uniformly over hours. Forecast inputs are
    PSEUDO-FORECASTS = future ERA5 accumulation x lognormal error, NOT archived forecasts available at the time.
    Experiments are therefore also run with forecast_mode='missing'. River-gauge input is unavailable (missing).
"""
from __future__ import annotations

from datetime import date

import numpy as np

from floodtask.data.real_event import ROOT_OUT
from floodtask.env.tasking_env import FloodTaskingEnv, N_TILES, tile_view
from floodtask.eval.metrics import boundary_f1, scores_from_confusion
from floodtask.hydro.simulator import EXT, NE, NT, SimParams, load_scene
from floodtask.utils import load_json


class ReplayEnv(FloodTaskingEnv):
    def __init__(self, cfg, sensor, scenes, event: str, seed=0, use_forecast=True):
        super().__init__(cfg, sensor, scenes, seed, use_forecast, hydro_obs=False)
        self.dir = ROOT_OUT / event
        self.era5 = load_json(self.dir / "era5_daily.json")
        self.start = date.fromisoformat(self.era5["dates"][0])

    def reset(self, spec=None):
        spec = spec or self._default_spec()
        self.spec, rng = spec, np.random.default_rng(spec.seed)
        self.rng = rng
        sc = self.sc = load_scene(spec.scene)
        m = np.load(self.dir / f"{spec.scene}_masks.npz")
        dates = [date.fromisoformat(str(d)) for d in m["dates"]]
        self.dates = dates
        self.P = len(dates)
        self.pass_t = np.array([(d - self.start).days * 24 + 12 for d in dates])   # ~mid-day UTC
        self.dt = float(np.mean(np.diff(self.pass_t))) if self.P > 1 else 24.0
        self.params = SimParams()
        hours = int(self.pass_t[-1] + 72 + 1)
        self._rain = self._era5_rain(hours)
        self._level = np.zeros((hours, sc["Z"]), np.float32)
        self._truth_acq = self._truth_eval = m["ref"].astype(bool)
        self._eval_valid = m["ref_valid"].astype(bool)
        self._obs_prob, self._obs_valid = m["obs_prob"], m["obs_valid"].astype(bool)
        self._feasible = np.stack([tile_view(v).reshape(N_TILES, -1).mean(1) >= 0.5 for v in self._obs_valid])
        for p in range(self.P):  # guarantee the budget can be spent: keep at least K feasible tiles
            if self._feasible[p].sum() < self.K:
                frac = tile_view(self._obs_valid[p]).reshape(N_TILES, -1).mean(1)
                self._feasible[p, np.argsort(-frac)[:self.K]] = True
        self._fail = np.zeros((self.P, N_TILES), bool)
        self._obs_seeds = rng.integers(2**31, size=(self.P, N_TILES))
        self.fmode = spec.forecast_mode or "operational"
        self._fparams = self._forecast_params(rng)
        self._fmiss = np.zeros(self.P, bool)
        self._gmiss = np.ones(self.P, bool)
        self._rmiss = np.zeros(self.P, bool)
        self._static = self._terrain_features(sc)
        self._reset_estimator()
        return self.observation()

    def _default_spec(self):
        from floodtask.env.tasking_env import EpisodeSpec
        return EpisodeSpec(self.scenes[0], 0)

    def _era5_rain(self, hours):
        days = [date.fromisoformat(d) for d in self.era5["dates"]]
        daily = dict(zip(days, self.era5["precip_mm"]))
        rain = np.zeros((hours, NE, NE), np.float32)
        for h in range(hours):
            d = date.fromordinal(self.start.toordinal() + h // 24)
            v = daily.get(d)
            rain[h] = (v if v is not None and np.isfinite(v) else 0.0) / 24.0
        return rain

    def _acquire(self, tile, p_idx):
        tp = tile_view(self._obs_prob[p_idx])[tile]
        tv = tile_view(self._obs_valid[p_idx])[tile]
        return tp.copy(), tv.copy()

    def evaluator_metrics(self, p_idx, light=False):
        truth, valid = self._truth_eval[p_idx], self._eval_valid[p_idx]
        pred = self.est_prob >= 0.5
        tp = int((pred & truth & valid).sum()); fp = int((pred & ~truth & valid).sum()); fn = int((~pred & truth & valid).sum())
        m = scores_from_confusion(tp, fp, fn, int(valid.sum()) - tp - fp - fn)
        if light:
            return m
        from floodtask.env.tasking_env import FLOOD_PRESENT_FRAC
        m.update(boundary_f1=boundary_f1(pred, truth, tol=1, valid=valid), true_flood_frac=float(truth[valid].mean()),
                 flood_present=bool(truth[valid].mean() >= FLOOD_PRESENT_FRAC),
                 false_alarm_frac=float((pred & ~truth & valid).sum() / max(valid.sum(), 1)),
                 pass_=p_idx, date=self.dates[p_idx].isoformat(), eval_valid_frac=float(valid.mean()),
                 acquired_tiles=int(self.acq_count.sum()), unique_tiles=int((self.acq_count > 0).sum()))
        m["pass"] = m.pop("pass_")
        return m
