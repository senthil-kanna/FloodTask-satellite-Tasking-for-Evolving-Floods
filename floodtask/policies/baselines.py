"""Conventional tasking baselines. Every operational policy sees ONLY the agent observation dict.

Common interface:  policy.reset(seed); policy.act(obs) -> tile index (must satisfy obs["mask"]).
Score-based policies pick the highest-scoring feasible, not-yet-selected tile; ties broken randomly.
"""
from __future__ import annotations

import numpy as np

from floodtask.env.tasking_env import GRID_CHANNELS, N_TILES

CH = {k: i for i, k in enumerate(GRID_CHANNELS)}


class Policy:
    name = "policy"
    operational = True

    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def score(self, obs) -> np.ndarray:
        raise NotImplementedError

    def act(self, obs, env=None) -> int:
        s = self.score(obs).astype(np.float64) + self.rng.random(N_TILES) * 1e-6
        s[~obs["mask"]] = -np.inf
        return int(np.argmax(s))


def g(obs, name):
    return obs["grid"][CH[name]].ravel()


class RandomPolicy(Policy):
    name = "random"

    def score(self, obs):
        return self.rng.random(N_TILES)


class StalestPolicy(Policy):
    """Uniform coverage: longest-unobserved tile first (never-observed tiles have age 1)."""
    name = "stalest"

    def score(self, obs):
        return g(obs, "obs_age")


class FloodedFirstPolicy(Policy):
    """Prioritise tiles currently mapped as flooded; fall back to flood-prone terrain, then staleness."""
    name = "flooded_first"

    def __init__(self, w_terrain=0.3, w_age=0.1):
        self.w_terrain, self.w_age = w_terrain, w_age

    def score(self, obs):
        return g(obs, "est_flood_frac") + self.w_terrain * g(obs, "hand_lt3") * g(obs, "never_observed") \
            + self.w_age * g(obs, "obs_age")


class ForecastPolicy(Policy):
    """Forecast-following: forecast rain x flood susceptibility; susceptibility x staleness if forecast missing."""
    name = "forecast"

    def __init__(self, w_susc=0.5, w_age=0.2):
        self.w_susc, self.w_age = w_susc, w_age

    def score(self, obs):
        susc = g(obs, "hand_lt3") + self.w_susc
        if g(obs, "fcst_missing")[0] > 0.5:
            return susc * g(obs, "obs_age")
        return (g(obs, "fcst_24h") + 0.5 * g(obs, "fcst_72h")) * susc + self.w_age * g(obs, "obs_age")


class GreedyUncertaintyPolicy(Policy):
    """Greedy expected-error heuristic: P(tile wrong) ~ susceptibility x uncertainty.
    uncertainty = 1 - (1 - sensor_entropy) * exp(-age_h / tau); susceptibility mixes terrain,
    current flood estimate and forecast rain. Weights tuned on validation scenes (scripts/tune_baselines.py)."""
    name = "greedy_unc"

    def __init__(self, tau_h=36.0, w_terrain=1.0, w_flood=1.0, w_rain=0.5, w0=0.05, horizon_h=144.0):
        self.tau_h, self.w_terrain, self.w_flood, self.w_rain, self.w0, self.horizon_h = tau_h, w_terrain, w_flood, w_rain, w0, horizon_h

    def score(self, obs):
        age_h = g(obs, "obs_age") * self.horizon_h
        never = g(obs, "never_observed") > 0.5
        unc = np.where(never, 1.0, 1 - (1 - g(obs, "sensor_uncertainty")) * np.exp(-age_h / self.tau_h))
        rain = 0 if g(obs, "fcst_missing")[0] > 0.5 else g(obs, "fcst_24h")
        susc = self.w0 + self.w_terrain * g(obs, "hand_lt3") + self.w_flood * g(obs, "est_flood_frac") + self.w_rain * rain * g(obs, "hand_lt3")
        return susc * unc


class OraclePolicy(Policy):
    """NON-OPERATIONAL upper reference: greedy on the true current map error (uses hidden ground truth)."""
    name = "oracle"
    operational = False

    def act(self, obs, env=None):
        s = env.oracle_tile_scores().astype(np.float64) + self.rng.random(N_TILES) * 1e-6
        s[~obs["mask"]] = -np.inf
        return int(np.argmax(s))


BASELINES = {c.name: c for c in [RandomPolicy, StalestPolicy, FloodedFirstPolicy, ForecastPolicy, GreedyUncertaintyPolicy, OraclePolicy]}
