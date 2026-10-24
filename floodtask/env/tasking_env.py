"""Partially observable multi-pass satellite tasking environment.

Information separation (enforced by tests/test_leakage.py):
  * hidden:    self._level, self._truth_*, self._rain (simulator state, including the future)
  * agent:     observation() -> {"grid", "vec", "mask"}  (only decision-time information)
  * evaluator: evaluator_metrics(), oracle_tile_scores()  (ground truth; analysis / training reward only)

One env step = one tile selection. A pass consists of `budget_per_pass` selections; acquisitions are
processed after the last selection of the pass and delivered `latency_passes` later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floodtask.data.scenes import SCENE, TILE
from floodtask.env.sensor import SensorParams, observe
from floodtask.eval.metrics import boundary_f1, scores_from_confusion
from floodtask.hydro.simulator import EXT, NE, NT, SimParams, flood_mask, load_scene, simulate, storm_field, upstream_mask

N_TILES = NT * NT
PAD = 3
GRID_CHANNELS = [
    "est_flood_frac", "est_mean_prob", "sensor_uncertainty", "obs_age", "never_observed", "last_valid_frac",
    "fcst_24h", "fcst_72h", "rain_obs_24h", "fcst_missing", "hand_mean", "hand_lt1", "hand_lt3",
    "log_acc", "drainage_frac", "feasible", "selected_this_pass", "last_acq_failed",
]
VEC_FEATURES = [
    "pass_frac", "budget_left_frac", "gauge_level", "gauge_change_24h", "gauge_missing", "upstream_fcst_24h",
    "upstream_fcst_72h", "fcst_missing", "rain_obs_missing", "latency",
]
FORECAST_CHANNELS = ["fcst_24h", "fcst_72h", "fcst_missing"]
FORECAST_VEC = ["upstream_fcst_24h", "upstream_fcst_72h", "fcst_missing"]
STALE_TAU_H = 48.0  # time scale for staleness in the uncertainty definition
FLOOD_PRESENT_FRAC = 0.001  # pass counts as "flood present" (IoU defined) if true flooded fraction >= 0.1%
IOU_EPS = int(round(FLOOD_PRESENT_FRAC * SCENE * SCENE))  # smoothing (cells) of the IoU difference reward


def tile_view(a: np.ndarray) -> np.ndarray:
    """[SCENE,SCENE] -> [N_TILES, TILE, TILE]"""
    return a.reshape(NT, TILE, NT, TILE).transpose(0, 2, 1, 3).reshape(N_TILES, TILE, TILE)


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


@dataclass
class EpisodeSpec:
    scene: str
    seed: int
    forecast_mode: str | None = None


class FloodTaskingEnv:
    def __init__(self, cfg: dict, sensor: SensorParams, scenes: list[str], seed: int = 0,
                 use_forecast: bool = True, hydro_obs: bool = True):
        self.cfg, self.sensor, self.scenes = cfg, sensor, scenes
        self.P, self.K = cfg["passes"], cfg["budget_per_pass"]
        self.dt = cfg["pass_interval_h"]
        self.use_forecast, self.hydro_obs = use_forecast, hydro_obs
        self.master = np.random.default_rng(seed)

    # ------------------------------------------------------------------ episode generation (hidden)
    def reset(self, spec: EpisodeSpec | None = None):
        if spec is None:
            spec = EpisodeSpec(self.scenes[self.master.integers(len(self.scenes))], int(self.master.integers(2**31)))
        self.spec = spec
        rng = self.rng = np.random.default_rng(spec.seed)
        sc = self.sc = load_scene(spec.scene)
        c, ph = self.cfg, self.cfg["physics"]
        u = lambda k: rng.uniform(*ph[k])
        p = SimParams(runoff_coef=u("runoff_coef"), residence_h=rng.uniform(*ph["residence_h"], sc["Z"]),
                      storm_total_mm=float(np.exp(rng.uniform(*np.log(ph["storm_total_mm"])))),
                      n_storms=int(rng.integers(ph["n_storms"][0], ph["n_storms"][1] + 1)),
                      ext_scale=float(np.exp(rng.uniform(*np.log(ph["ext_scale"])))), ext_lag_h=u("ext_lag_h"),
                      loss_mm=u("loss_mm"))
        self.params = p
        self.pass_t = c["spinup_h"] + self.dt * np.arange(self.P)
        hours = int(self.pass_t[-1] + 72 + c["eval_delay_h"] + 1)
        self._rain = storm_field(rng, hours, p)
        self._level = simulate(sc, self._rain, p)
        self._truth_acq = np.stack([flood_mask(sc, self._level[t]) for t in self.pass_t])
        self._truth_eval = np.stack([flood_mask(sc, self._level[t + c["eval_delay_h"]]) for t in self.pass_t])
        self._truth_acq_pad = np.pad(self._truth_acq, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="edge")
        # acquisition feasibility and failures are exogenous: pre-sampled so every policy faces the same ones
        self._feasible = np.stack([self._swath(rng) for _ in range(self.P)])
        self._fail = rng.random((self.P, N_TILES)) < c["acq_fail_prob"]
        self._obs_seeds = rng.integers(2**31, size=(self.P, N_TILES))
        fc = c["forecast"]
        self.fmode = spec.forecast_mode or fc["mode"]
        self._fparams = self._forecast_params(rng)
        self._fmiss = rng.random(self.P) < (fc["forecast_missing_prob"] if self.fmode == "operational" else 0.0)
        self._gmiss = rng.random(self.P) < fc["gauge_missing_prob"]
        self._rmiss = rng.random(self.P) < fc["rain_obs_missing_prob"]
        self._static = self._terrain_features(sc)
        self._reset_estimator()
        return self.observation()

    def _prior_logit_shift(self, sc) -> np.ndarray | None:
        """Shared-estimator terrain prior: logit P(water|HAND) - logit pi_train (prior-shift correction of the
        segmentation output). Fitted on training events (scripts/fit_hand_prior.py). None = use raw probabilities."""
        est = self.cfg.get("estimator", {}).get("kind", "latest")
        if est == "latest":
            return None
        hp = self.cfg["hand_prior"]
        lg = lambda q: np.log(q / (1 - q))
        return (hp["intercept"] + hp["slope_log1p_hand"] * np.log1p(sc["hand"]) - lg(hp["pi_train"])).astype(np.float32)

    def _fuse(self, prob, tile):
        if self._shift is None:
            return prob
        r, c = divmod(tile, NT)
        sh = self._shift[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
        p = np.clip(prob, 1e-6, 1 - 1e-6)
        return (1 / (1 + np.exp(-(np.log(p / (1 - p)) + sh)))).astype(np.float32)

    def _reset_estimator(self):
        self._conf_key = None
        self._shift = self._prior_logit_shift(self.sc)
        # agent-side estimator state (shared map-fusion module, identical for every policy)
        self.est_prob = np.zeros((SCENE, SCENE), np.float32)       # prior: not flooded
        self.est_valid = np.zeros((SCENE, SCENE), bool)
        self.last_obs_t = np.full(N_TILES, -np.inf)
        self.sensor_unc = np.ones(N_TILES, np.float32)
        self.last_valid_frac = np.zeros(N_TILES, np.float32)
        self.last_failed = np.zeros(N_TILES, bool)
        self.pending: list[tuple] = []   # (deliver_pass, tile, prob, valid, t_acq)
        self.pass_idx, self.k = 0, 0
        self.selected = np.zeros(N_TILES, bool)
        self.history = []          # audit trail
        self.acq_count = np.zeros(N_TILES, int)
        self._forecast_cache = {}

    def _swath(self, rng):
        if rng.random() < self.cfg["full_swath_prob"]:
            return np.ones(N_TILES, bool)
        w = int(rng.integers(self.cfg["swath_tiles"][0], self.cfg["swath_tiles"][1] + 1))
        off = int(rng.integers(0, NT - w + 1))
        m = np.zeros((NT, NT), bool)
        if rng.random() < 0.5:
            m[:, off:off + w] = True   # ascending-like access band
        else:
            m[off:off + w, :] = True
        return m.ravel()

    def _forecast_params(self, rng):
        m = self.fmode
        sig = {"accurate": (0.1, 0.1), "operational": (0.3, 0.5), "noisy": (0.8, 1.2)}.get(m, (0.3, 0.5))
        bias = {"biased_low": np.log(0.4), "biased_high": np.log(2.5)}.get(m, rng.normal(0, 0.3) if m == "operational" else 0.0)
        if m == "displaced":
            r, a = rng.uniform(4, 6), rng.uniform(0, 2 * np.pi)
            shift = (int(round(r * np.sin(a))), int(round(r * np.cos(a))))
        elif m == "operational":
            shift = tuple(int(x) for x in rng.integers(-2, 3, 2))
        else:
            shift = (0, 0)
        return {"sig": sig, "bias": bias, "shift": shift}

    @staticmethod
    def _terrain_features(sc):
        hv = tile_view(sc["hand"])
        acc = tile_view(sc["acc"]).reshape(N_TILES, -1).max(1)
        return {
            "hand_mean": np.clip(hv.reshape(N_TILES, -1).mean(1) / 20, 0, 1),
            "hand_lt1": (hv < 1).reshape(N_TILES, -1).mean(1),
            "hand_lt3": (hv < 3).reshape(N_TILES, -1).mean(1),
            "log_acc": np.log(acc) / np.log(256 * 256),
            "drainage_frac": tile_view(sc["drainage"]).reshape(N_TILES, -1).mean(1),
        }

    # ------------------------------------------------------------------ decision-time information
    def _forecast(self, p_idx):
        """Rain accumulation forecast issued at pass time (SYNTHETIC: truth + documented error model)."""
        if p_idx in self._forecast_cache:
            return self._forecast_cache[p_idx]
        t = int(self.pass_t[p_idx])
        fr = np.random.default_rng(self._obs_seeds[p_idx, 0] + 7)
        out = {}
        for h, sig in zip((24, 72), self._fparams["sig"]):
            acc = self._rain[t:t + h].sum(0)                          # truth on extended grid [NE,NE]
            dy, dx = self._fparams["shift"]
            win = acc[EXT + dy:EXT + dy + NT, EXT + dx:EXT + dx + NT]
            up = acc[upstream_mask(self.sc["upstream_side"])].mean()
            if self.fmode == "uninformative":
                clim = np.mean(self.cfg["physics"]["storm_total_mm"]) * h / (self.P * self.dt)
                win, up = np.full_like(win, clim), clim
            else:
                noise = fr.normal(0, sig, win.shape)
                win = win * np.exp(self._fparams["bias"] + noise)
                up = up * np.exp(self._fparams["bias"] + fr.normal(0, sig / 2))
            out[h] = (win.ravel(), up)
        missing = self.fmode == "missing" or bool(self._fmiss[p_idx]) or not self.use_forecast
        out["missing"] = missing
        self._forecast_cache[p_idx] = out
        return out

    def observation(self):
        p = min(self.pass_idx, self.P - 1)
        t = self.pass_t[p]
        tv = tile_view(self.est_prob).reshape(N_TILES, -1)
        age = np.where(np.isfinite(self.last_obs_t), (t - self.last_obs_t) / (self.P * self.dt), 1.0)
        f = self._forecast(p)
        fm = f["missing"]
        z = np.zeros(N_TILES, np.float32)
        rain24 = self._rain[max(int(t) - 24, 0):int(t), EXT:EXT + NT, EXT:EXT + NT].sum(0).ravel()
        rain24 = rain24 * np.exp(np.random.default_rng(self._obs_seeds[p, 1]).normal(0, 0.2, N_TILES))
        rmiss = bool(self._rmiss[p]) or not self.hydro_obs
        g = {
            "est_flood_frac": (tv >= 0.5).mean(1), "est_mean_prob": tv.mean(1), "sensor_uncertainty": self.sensor_unc,
            "obs_age": np.clip(age, 0, 1), "never_observed": ~np.isfinite(self.last_obs_t),
            "last_valid_frac": self.last_valid_frac,
            "fcst_24h": z if fm else np.log1p(f[24][0]) / 5, "fcst_72h": z if fm else np.log1p(f[72][0]) / 5,
            "rain_obs_24h": z if rmiss else np.log1p(rain24) / 5, "fcst_missing": np.full(N_TILES, float(fm)),
            **self._static, "feasible": self._feasible[p], "selected_this_pass": self.selected,
            "last_acq_failed": self.last_failed,
        }
        grid = np.stack([np.asarray(g[k], np.float32) for k in GRID_CHANNELS]).reshape(len(GRID_CHANNELS), NT, NT)
        gmiss = bool(self._gmiss[p]) or not self.hydro_obs
        lvl = self._level[int(t), self.sc["main_zone"]] + np.random.default_rng(self._obs_seeds[p, 2]).normal(0, 0.05)
        lvl24 = self._level[max(int(t) - 24, 0), self.sc["main_zone"]]
        vec = np.array([
            self.pass_idx / self.P, (self.K - self.k) / self.K,
            0.0 if gmiss else lvl / 5, 0.0 if gmiss else (lvl - lvl24) / 2, float(gmiss),
            0.0 if fm else np.log1p(f[24][1]) / 5, 0.0 if fm else np.log1p(f[72][1]) / 5, float(fm), float(rmiss),
            self.cfg["latency_passes"] / 2,
        ], np.float32)
        mask = self._feasible[p] & ~self.selected
        return {"grid": grid, "vec": vec, "mask": mask}

    # ------------------------------------------------------------------ dynamics
    def _acquire(self, tile, p_idx):
        r, c = divmod(tile, NT)
        win = self._truth_acq_pad[p_idx, r * TILE:(r + 1) * TILE + 2 * PAD, c * TILE:(c + 1) * TILE + 2 * PAD]
        return observe(win, PAD, self.sensor, np.random.default_rng(self._obs_seeds[p_idx, tile]))

    def _tile_unc(self, tile, t):
        if not np.isfinite(self.last_obs_t[tile]):
            return 1.0
        stale = 1 - np.exp(-(t - self.last_obs_t[tile]) / STALE_TAU_H)
        return 1 - (1 - self.sensor_unc[tile]) * (1 - stale)

    def _integrate(self, tile, prob, valid, t_acq):
        r, c = divmod(tile, NT)
        sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
        cur = self.est_prob[sl]
        self.est_prob[sl] = np.where(valid, prob, cur)      # nodata keeps the previous estimate
        self.est_valid[sl] |= valid
        self.last_obs_t[tile] = t_acq
        self.sensor_unc[tile] = float(binary_entropy(prob[valid]).mean()) if valid.any() else 1.0
        self.last_valid_frac[tile] = valid.mean()

    def step(self, action: int):
        p = self.pass_idx
        if not (0 <= action < N_TILES) or not self._feasible[p, action] or self.selected[action]:
            raise ValueError(f"infeasible action {action} at pass {p}")
        t = self.pass_t[p]
        self.selected[action] = True
        failed = bool(self._fail[p, action])
        rcfg = self.cfg["reward"]
        reward = 0.0
        if not failed:
            prob, valid = self._acquire(action, p)
            prob = self._fuse(prob, action)
            dp = min(p + self.cfg["latency_passes"], self.P - 1)
            self.pending.append((p + self.cfg["latency_passes"], action, prob, valid, t))
            truth = tile_view(self._truth_eval[dp])[action]
            cur = tile_view(self.est_prob)[action] >= 0.5
            new = np.where(valid, prob >= 0.5, cur)
            if rcfg["kind"] == "err":  # difference reward: (cost-weighted) error reduction in this tile at delivery
                w = rcfg.get("fn_weight", 1.0)   # cost of a missed flood cell relative to a false alarm
                cost = lambda m: w * (~m & truth).sum() + (m & ~truth).sum()
                reward = float(cost(cur) - cost(new)) / (TILE * TILE)
            elif rcfg["kind"] == "iou_diff":
                # difference reward on the evaluation metric: change of GLOBAL map IoU (at delivery time) caused by
                # this tile's observation alone, rest of the map held fixed. Scaled to IoU percentage points.
                tp, fp, fn = self._global_confusion(dp)
                d = lambda m: (int((m & truth).sum()), int((m & ~truth).sum()), int((~m & truth).sum()))
                (tp0, fp0, fn0), (tp1, fp1, fn1) = d(cur), d(new)
                iou = lambda a, b, c: (a + IOU_EPS) / (a + b + c + IOU_EPS)   # smoothed: defined when flood is absent
                reward = 100 * (iou(tp - tp0 + tp1, fp - fp0 + fp1, fn - fn0 + fn1) - iou(tp, fp, fn))
            if rcfg["unc_coef"] > 0:
                u_new = float(binary_entropy(prob[valid]).mean()) if valid.any() else 1.0
                reward += rcfg["unc_coef"] * (self._tile_unc(action, t) - u_new)
        self.acq_count[action] += 1
        self.history.append({"pass": int(p), "slot": int(self.k), "tile": int(action), "failed": failed,
                             "t_h": float(t), "age_before_h": float(t - self.last_obs_t[action])
                             if np.isfinite(self.last_obs_t[action]) else None})
        self.last_failed[action] = failed
        self.k += 1
        info = {}
        if self.k == self.K:
            info = self._end_pass(rcfg)
            if rcfg["kind"] == "iou":
                reward += info["pass_metrics"]["reward_iou_delta"]
        done = self.pass_idx >= self.P
        obs = None if done else self.observation()
        return obs, reward, done, info

    def _global_confusion(self, dp):
        """TP/FP/FN of the current estimate vs truth at pass dp; cached while the estimate is unchanged (within a pass)."""
        key = (self.pass_idx, dp)
        if getattr(self, "_conf_key", None) != key:
            pred, truth = self.est_prob >= 0.5, self._truth_eval[dp]
            self._conf = (int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum()))
            self._conf_key = key
        return self._conf

    def _end_pass(self, rcfg):
        p = self.pass_idx
        before = self.evaluator_metrics(p, light=True)["iou"] if rcfg["kind"] == "iou" else 0.0
        due = [x for x in self.pending if x[0] <= p]
        self.pending = [x for x in self.pending if x[0] > p]
        for _, tile, prob, valid, t_acq in due:
            self._integrate(tile, prob, valid, t_acq)
        self._conf_key = None
        metrics = self.evaluator_metrics(p)
        metrics["reward_iou_delta"] = metrics["iou"] - before if rcfg["kind"] == "iou" else 0.0
        self.pass_idx += 1
        self.k = 0
        self.selected[:] = False
        return {"pass_metrics": metrics}

    # ------------------------------------------------------------------ map product (agent-side, no ground truth)
    UNKNOWN = 255

    def map_product(self, t_now: float | None = None) -> dict:
        """Deliverable map layers from the shared estimator (decision-time information only).
        flood_class: 0 dry, 1 flooded, 255 unknown (never validly observed). Metrics treat unknown as dry,
        so this does not change evaluation; it makes blind spots explicit for users.
        confidence: 0-100, 100 * (1 - binary entropy of the fused probability); 0 where unknown.
        age_h: hours since the cell's tile was last delivered; 65535 where never observed."""
        t_now = self.pass_t[min(self.pass_idx, self.P - 1)] if t_now is None else t_now
        cls = np.where(self.est_prob >= 0.5, 1, 0).astype(np.uint8)
        cls[~self.est_valid] = self.UNKNOWN
        conf = np.where(self.est_valid, 100 * (1 - binary_entropy(self.est_prob)), 0).astype(np.uint8)
        age_tile = np.where(np.isfinite(self.last_obs_t), np.clip(t_now - self.last_obs_t, 0, 65534), 65535)
        age = np.kron(age_tile.reshape(NT, NT), np.ones((TILE, TILE))).astype(np.uint16)
        return {"flood_class": cls, "confidence": conf, "age_h": age}

    # ------------------------------------------------------------------ evaluator / oracle (ground truth)
    def evaluator_metrics(self, p_idx: int, light: bool = False) -> dict:
        truth = self._truth_eval[p_idx]
        pred = self.est_prob >= 0.5
        tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
        m = scores_from_confusion(tp, fp, fn, truth.size - tp - fp - fn)
        if light:
            return m
        m["boundary_f1"] = boundary_f1(pred, truth, tol=1)
        m["true_flood_frac"] = float(truth.mean())
        m["flood_present"] = bool(truth.mean() >= FLOOD_PRESENT_FRAC)
        m["false_alarm_frac"] = float((pred & ~truth).mean())
        m["pass"] = p_idx
        m["acquired_tiles"] = int(self.acq_count.sum())
        m["unique_tiles"] = int((self.acq_count > 0).sum())
        m["unique_flooded_tiles_seen"] = int(((self.acq_count > 0) & (tile_view(truth).reshape(N_TILES, -1).mean(1) > 0.01)).sum())
        m["flooded_tiles"] = int((tile_view(truth).reshape(N_TILES, -1).mean(1) > 0.01).sum())
        return m

    def oracle_tile_scores(self) -> np.ndarray:
        """NON-OPERATIONAL: current map error per tile against the truth at this pass's evaluation time."""
        p = self.pass_idx
        dp = min(p + self.cfg["latency_passes"], self.P - 1)
        err = (tile_view(self.est_prob >= 0.5) != tile_view(self._truth_eval[dp])).reshape(N_TILES, -1).sum(1)
        return err.astype(np.float32)
