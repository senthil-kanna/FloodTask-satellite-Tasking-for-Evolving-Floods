"""Build the data for the map viewer (results/viewer/index.html) by replaying the deployed policies.

  python scripts/build_viewer_data.py            # writes results/viewer/data/
Events:
  - assam2016, assam2022: real Sentinel-1 replays (ReplayEnv), forecast 'operational' (ERA5 pseudo-forecast), episode seed 0;
    truth = model-derived reference mask (0 dry, 1 water, 2 excluded).
  - sim_bolivia, sim_india, sim_mekong: held-out simulated test scenes, the first core-suite episode of each scene
    (same seed as results/experiments/core); truth = hidden simulated flood state at evaluation time.
Policies: recurrent PPO (best-validation seed, as scripts/export_maps.py) and tuned greedy uncertainty.
Per pass: map product (flood class, confidence decile), the 16 selected tiles, truth, and the flood outlook
(floodtask/forecast/outlook.py) issued from the PPO map for +4, +6, +12 and +24 h; for real events the outlook is also scored
against the next reference date.
File format: <event>_<policy>.b64.txt / <event>_truth.b64.txt / <event>_outlook.b64.txt = base64(zlib(bytes)), one byte
per 60 m cell, 192 x 192 per pass, passes concatenated scene by scene. Map byte: 255 unknown, else class*10 + confidence
decile. Outlook: two planes per pass, byte = round(10*P_a)*11 + round(10*P_b) for (a, b) = (4, 6) then (12, 24). scenes.json holds offsets, dates, plans and scores.
"""
from __future__ import annotations

import base64
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rasterio.warp import transform_bounds

from floodtask.data.real_event import ROOT_OUT, scenes as real_scenes
from floodtask.data.scenes import CELL, SCENE
from floodtask.env.factory import env_config, make_env, scene_names, sensor_params, specs, VARIANTS
from floodtask.env.replay_env import ReplayEnv
from floodtask.env.tasking_env import EpisodeSpec
from floodtask.forecast.outlook import flood_outlook
from floodtask.hydro.simulator import flood_mask
from floodtask.utils import DATA, load_json, RESULTS

sys.path.insert(0, str(RESULTS.parent / "scripts"))
OUT = RESULTS / "viewer" / "data"
POLS = ("ppo_recurrent", "greedy_unc")
HZ = (4, 6, 12, 24)
SIM_EVENTS = {"sim_bolivia": "Bolivia", "sim_india": "India", "sim_mekong": "Mekong"}
META = {s["scene"]: s for s in load_json(DATA / "processed" / "scenes.json")}


def policy(name):
    import torch
    from export_maps import ScoringRL, best_seed
    from run_experiments import tuned
    from floodtask.rl.model import ActorCritic
    if name == "greedy_unc":
        return tuned("greedy_unc")
    seed = best_seed("recurrent")
    m = ActorCritic(recurrent=VARIANTS["recurrent"]["recurrent"])
    m.load_state_dict(torch.load(RESULTS / "rl" / "recurrent" / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True))
    pol = ScoringRL(m, "cpu", name="ppo_recurrent")
    pol.seed_used = seed
    return pol


def make(event, scene, use_forecast=True):
    if event.startswith("sim_"):
        return make_env("test")
    return ReplayEnv(env_config(), sensor_params(), [scene], event, use_forecast=use_forecast)


def r4(v):
    return None if v is None else round(float(v), 4)


def run_scene(args):
    event, scene, spec = args
    real = not event.startswith("sim_")
    res = {"scene": scene, "policies": {}, "maps": {}}
    for name in POLS:
        env, pol = make(event, scene), policy(name)
        obs = env.reset(spec)
        pol.reset(0)
        maps, truth, outlook, passes, plan, done = [], [], [], [], [], False
        while not done:
            p = env.pass_idx
            a = pol.act(obs, None if pol.operational else env)
            prob = getattr(pol, "last_prob", None)
            obs, _, done, info = env.step(a)
            h = env.history[-1]
            plan.append([int(a), h["slot"] + 1, r4(prob), int(h["failed"]), None if h["age_before_h"] is None else int(h["age_before_h"])])
            if "pass_metrics" not in info:
                continue
            pm, t = info["pass_metrics"], float(env.pass_t[p])
            mp = env.map_product(t)
            cls, conf = mp["flood_class"], mp["confidence"]
            maps.append(np.where(cls == 255, 255, cls.astype(np.int32) * 10 + np.minimum(conf // 10, 9)).astype(np.uint8))
            age = mp["age_h"][6::12, 6::12].astype(np.int64).ravel()
            row = {"t_h": t, "date": env.dates[p].isoformat() if real else None,
                   "iou": r4(pm["iou"]) if pm.get("flood_present", True) else None, "precision": r4(pm["precision"]),
                   "recall": r4(pm["recall"]), "true_flood_frac": r4(pm["true_flood_frac"]),
                   "flood_frac": r4((cls == 1).mean()), "known_frac": r4((cls != 255).mean()),
                   "age_h": [int(v) if v != 65535 else -1 for v in age], "plan": plan}
            plan = []
            if name == "ppo_recurrent":
                tr = env._truth_eval[p]
                truth.append(np.where(env._eval_valid[p], tr, 2).astype(np.uint8) if real else tr.astype(np.uint8))
                lead = int(env.pass_t[p + 1] - env.pass_t[p]) if real and p + 1 < env.P else None
                o = flood_outlook(env, p, HZ + ((lead,) if lead and lead not in HZ else ()), seed=spec.seed)
                pair = lambda a, b: (np.rint(o[a] * 10) * 11 + np.rint(o[b] * 10)).astype(np.uint8)
                outlook.append(np.stack([pair(4, 6), pair(12, 24)]))
                row["outlook_ess"] = round(o["ess"], 1)
                row["outlook_flood_frac"] = {str(h): r4((o[h] >= 0.5).mean()) for h in HZ}
                if real and lead:
                    nxt, valid = env._truth_eval[p + 1], env._eval_valid[p + 1]
                    def sc(prob):
                        pr = prob >= 0.5
                        tp = int((pr & nxt & valid).sum()); fp = int((pr & ~nxt & valid).sum()); fn = int((~pr & nxt & valid).sum())
                        return r4(tp / max(tp + fp + fn, 1)) if (nxt & valid).sum() >= 37 else None
                    row["next"] = {"lead_h": lead, "iou_outlook": sc(o[lead]), "iou_persistence": sc(env.est_prob)}
                elif not real:
                    row["next"] = {}
                    for hh in HZ:
                        fut = flood_mask(env.sc, env._level[int(t) + hh])
                        if fut.mean() >= 0.001:
                            for k, prob in (("outlook", o[hh]), ("persistence", env.est_prob)):
                                pr = prob >= 0.5
                                tp = int((pr & fut).sum()); fp = int((pr & ~fut).sum()); fn = int((~pr & fut).sum())
                                row["next"][f"iou_{k}_{hh}"] = r4(tp / max(tp + fp + fn, 1))
            passes.append(row)
        res["policies"][name] = {"passes": passes}
        res["maps"][name] = np.stack(maps)
        if name == "ppo_recurrent":
            res["maps"]["truth"], res["maps"]["outlook"] = np.stack(truth), np.stack(outlook)
            res["seed_used"] = getattr(pol, "seed_used", None)
    return res


def bounds(scene):
    m = META[scene]
    tf = m["transform"]
    x0, y0 = tf[2], tf[5]
    return [round(v, 5) for v in transform_bounds(f"EPSG:{m['epsg']}", "EPSG:4326", x0, y0 - SCENE * CELL, x0 + SCENE * CELL, y0)]


def terrain(scene):
    z = np.load(DATA / "processed" / "scenes" / f"{scene}.npz")
    dem = z["dem"].astype(np.float64)
    gy, gx = np.gradient(dem, CELL)
    slope, aspect = np.arctan(np.hypot(gx, gy) * 3), np.arctan2(-gx, gy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    hs = np.clip((hs - hs.min()) / max(hs.max() - hs.min(), 1e-9) * 250, 0, 250).astype(np.uint8)
    hs[z["drainage"]] = 255
    return hs


def write(name, arrays):
    data = base64.b64encode(zlib.compress(np.concatenate(arrays).tobytes(), 9)).decode()
    (OUT / f"{name}.b64.txt").write_text(data)
    print(f"  {name}: {len(data) / 1e6:.2f} MB")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for ev in ("assam2016", "assam2022"):
        names = [s["scene"] for s in real_scenes(ev) if (ROOT_OUT / ev / f"{s['scene']}_masks.npz").exists()]
        jobs[ev] = [(ev, s, EpisodeSpec(s, 0, "operational")) for s in names]
    test = scene_names("test")
    for key, country in SIM_EVENTS.items():
        jobs[key] = [(key, sp.scene, sp) for sp in specs("test", len(test)) if META[sp.scene]["event"] == country]
    meta = {"grid": {"cells": SCENE, "tiles": 16, "tile_cells": 12, "cell_m": CELL}, "events": {}}
    used = set()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for key, js in jobs.items():
            print(key, len(js), "scenes")
            results = list(ex.map(run_scene, js))
            results.sort(key=lambda r: int(r["scene"].split("_")[1]))
            E = {"kind": "sim" if key.startswith("sim_") else "real", "country": SIM_EVENTS.get(key, "India"),
                 "policies": list(POLS), "scenes": {}}
            stacks = {k: [] for k in (*POLS, "truth", "outlook")}
            off = 0
            for r in results:
                n = len(r["policies"]["ppo_recurrent"]["passes"])
                S = {"bounds": bounds(r["scene"]), "offset": off, "n_passes": n, "policies": r["policies"],
                     "ppo_seed": r.get("seed_used")}
                if key.startswith("sim_"):
                    S["episode_seed"] = next(j[2].seed for j in js if j[1] == r["scene"])
                E["scenes"][r["scene"]] = S
                for k in stacks:
                    stacks[k].append(r["maps"][k])
                off += n
                used.add(r["scene"])
            for k, arr in stacks.items():
                write(f"{key}_{k}", arr)
            meta["events"][key] = E
    order = sorted(used, key=lambda s: (META[s]["event"], int(s.split("_")[1])))
    meta["terrain_scenes"] = order
    write("terrain", [terrain(s) for s in order])
    (OUT / "scenes.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("scenes.json", (OUT / "scenes.json").stat().st_size / 1e6, "MB")
    # outlook skill on the real events: outlook vs persistence at the next reference date
    rows = [(ev, s, p["next"]) for ev in ("assam2016", "assam2022") for s, S in meta["events"][ev]["scenes"].items()
            for p in S["policies"]["ppo_recurrent"]["passes"] if p.get("next") and p["next"].get("iou_outlook") is not None]
    summ = {}
    for ev in ("assam2016", "assam2022"):
        r = [x[2] for x in rows if x[0] == ev]
        if r:
            summ[ev] = {"n_cases": len(r), "lead_h_median": float(np.median([x["lead_h"] for x in r])),
                        "iou_outlook": float(np.mean([x["iou_outlook"] for x in r])),
                        "iou_persistence": float(np.mean([x["iou_persistence"] for x in r])),
                        "outlook_better_share": float(np.mean([x["iou_outlook"] > x["iou_persistence"] for x in r]))}
    print(summ)
    json.dump({"real_event_next_date": summ, "note": "outlook issued from the PPO map at each pass, scored at the next "
               "reference date on reference-valid cells; persistence = the current map unchanged"},
              open(RESULTS / "outlook_real.json", "w"), indent=1)


if __name__ == "__main__":
    main()
