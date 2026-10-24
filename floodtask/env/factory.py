"""Environment construction, experiment variants and fixed episode specifications."""
from __future__ import annotations

import copy

from floodtask.env.sensor import PERFECT, SensorParams
from floodtask.env.tasking_env import EpisodeSpec, FloodTaskingEnv
from floodtask.utils import DATA, load_json, load_yaml, RESULTS, ROOT

VARIANTS = {
    "recurrent":            {"recurrent": True,  "use_forecast": True,  "reward": {"kind": "err", "fn_weight": 3.0, "unc_coef": 0.0}},
    "feedforward":          {"recurrent": False, "use_forecast": True,  "reward": {"kind": "err", "fn_weight": 3.0, "unc_coef": 0.0}},
    "recurrent_noforecast": {"recurrent": True,  "use_forecast": False, "reward": {"kind": "err", "fn_weight": 3.0, "unc_coef": 0.0}},
    "recurrent_unc":        {"recurrent": True,  "use_forecast": True,  "reward": {"kind": "err", "fn_weight": 3.0, "unc_coef": 0.1}},
    "recurrent_iou":        {"recurrent": True,  "use_forecast": True,  "reward": {"kind": "iou", "fn_weight": 3.0, "unc_coef": 0.0}},
}
SPLIT_SEED_BASE = {"train": 0, "val": 10_000, "test": 20_000}


def scene_names(split: str) -> list[str]:
    """Split membership is always derived from configs/data.yaml (by event), never from cached metadata."""
    events = set(load_yaml(ROOT / "configs" / "data.yaml")["split"][split])
    return sorted(s["scene"] for s in load_json(DATA / "processed" / "scenes.json") if s["event"] in events)


def sensor_params(kind: str = "calibrated") -> SensorParams:
    if kind == "perfect":
        return PERFECT
    path = RESULTS / "sensor_model.json"
    sp = SensorParams(**load_json(path)["params"]) if path.exists() else SensorParams()
    if kind == "degraded":  # pessimistic segmentation: weaker signal, more noise
        sp = SensorParams(a=sp.a * 0.6, bias=sp.bias - 0.3, sigma=sp.sigma * 1.5, corr=sp.corr, blur=sp.blur * 1.5,
                          nodata_tile_prob=min(1.0, sp.nodata_tile_prob * 3))
    return sp


def env_config(overrides: dict | None = None) -> dict:
    cfg = load_yaml(ROOT / "configs" / "env.yaml")
    cal = RESULTS / "sim_calibration.json"
    if cal.exists():
        cfg["physics"].update(load_json(cal)["physics"])
    hp = RESULTS / "hand_prior.json"
    if hp.exists():
        cfg["hand_prior"] = load_json(hp)
    elif cfg.get("estimator", {}).get("kind", "latest") != "latest":
        raise FileNotFoundError("estimator needs results/hand_prior.json (run scripts/fit_hand_prior.py)")
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            cfg[k] = {**cfg.get(k, {}), **v}
        else:
            cfg[k] = v
    return cfg


def make_env(split: str, variant: str = "recurrent", seed: int = 0, sensor: str = "calibrated",
             overrides: dict | None = None, hydro_obs: bool = True) -> FloodTaskingEnv:
    v = VARIANTS[variant]
    cfg = env_config({"reward": v["reward"], **(overrides or {})})
    return FloodTaskingEnv(cfg, sensor_params(sensor), scene_names(split), seed=seed,
                           use_forecast=v["use_forecast"], hydro_obs=hydro_obs)


def specs(split: str, n: int, forecast_mode: str | None = None, offset: int = 0) -> list[EpisodeSpec]:
    names = scene_names(split)
    base = SPLIT_SEED_BASE[split] + offset
    return [EpisodeSpec(names[i % len(names)], base + i, forecast_mode) for i in range(n)]


def val_specs(n: int) -> list[EpisodeSpec]:
    return specs("val", n)
