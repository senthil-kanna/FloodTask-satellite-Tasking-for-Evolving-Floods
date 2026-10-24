"""Apply a frozen tasking policy to a real-event replay and export georeferenced products per acquisition date.

  python scripts/export_maps.py --event assam2016 [--policy ppo:recurrent:best] [--scenes India_00 India_01]
  python scripts/export_maps.py --event assam2016 --scene India_00 --plan-date 2016-08-12   # one plan, printed

Outputs in results/products/<event>/<policy>/<scene>/:
  <date>_map.tif   3 bands (uint16): 1 flood class (0 dry, 1 flood, 255 unknown), 2 confidence 0-100, 3 data age (h)
  <date>_plan.geojson  tiles tasked for that pass (rank, selection probability, failed flag, lon/lat polygon)
The map on <date> includes the acquisitions of that date (delivered before the 3 h product time).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform

from floodtask.data.real_event import scenes as event_scenes
from floodtask.data.scenes import CELL
from floodtask.env.factory import env_config, sensor_params, VARIANTS
from floodtask.env.replay_env import ReplayEnv
from floodtask.env.tasking_env import EpisodeSpec, TILE
from floodtask.eval.runner import RLPolicy
from floodtask.hydro.simulator import NT
from floodtask.rl.model import ActorCritic
from floodtask.utils import load_json, provenance, RESULTS, save_json

import sys
sys.path.insert(0, str(RESULTS.parent / "scripts"))
from run_experiments import tuned  # noqa: E402


def best_seed(variant: str) -> int:
    """Deployed checkpoint = seed with the best VALIDATION score (never test)."""
    logs = {int(p.parent.name[4:]): load_json(p)["best_val_mean_iou"] for p in (RESULTS / "rl" / variant).glob("seed*/train_log.json")}
    return max(logs, key=logs.get)


class ScoringRL(RLPolicy):
    """RLPolicy that also records the selection probability of each chosen tile."""
    @torch.no_grad()
    def act(self, obs, env=None):
        t = lambda a, dt: torch.as_tensor(a, dtype=dt)[None, None]
        logits, _, self.state = self.model(t(obs["grid"], torch.float32), t(obs["vec"], torch.float32),
                                           t(obs["mask"], torch.bool), self.state)
        p = torch.softmax(logits[0, 0], 0)
        a = int(torch.argmax(p))
        self.last_prob = float(p[a])
        return a


def make_policy(spec: str):
    if spec.startswith("ppo:"):
        _, variant, seed = spec.split(":")
        seed = best_seed(variant) if seed == "best" else int(seed)
        m = ActorCritic(recurrent=VARIANTS[variant]["recurrent"])
        m.load_state_dict(torch.load(RESULTS / "rl" / variant / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True))
        return ScoringRL(m, "cpu", name=f"ppo_{variant}_seed{seed}"), variant
    return tuned(spec), "recurrent"


def tile_polygon(meta, tile):
    r, c = divmod(tile, NT)
    x0, y0 = meta["transform"][2], meta["transform"][5]
    xs = [x0 + c * TILE * CELL, x0 + (c + 1) * TILE * CELL]
    ys = [y0 - r * TILE * CELL, y0 - (r + 1) * TILE * CELL]
    ring_x, ring_y = [xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]]
    lon, lat = warp_transform(f"EPSG:{meta['epsg']}", "EPSG:4326", ring_x, ring_y)
    return [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]


def run_scene(event, scene, meta, policy, variant, out_root, plan_date=None, forecast_mode="operational"):
    env = ReplayEnv(env_config(), sensor_params(), [scene], event, use_forecast=VARIANTS[variant]["use_forecast"])
    obs = env.reset(EpisodeSpec(scene, 0, forecast_mode))
    policy.reset(0)
    out = out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    tf = Affine(CELL, 0, meta["transform"][2], 0, -CELL, meta["transform"][5])
    done, plans = False, []
    while not done:
        p, date = env.pass_idx, env.dates[env.pass_idx].isoformat()
        a = policy.act(obs, None if policy.operational else env)
        feats = {"tile": a, "rank": env.k + 1, "selection_prob": getattr(policy, "last_prob", None)}
        obs, _, done, info = env.step(a)
        h = env.history[-1]
        plans.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [tile_polygon(meta, a)]},
                      "properties": {**feats, "date": date, "pass": p + 1, "failed": h["failed"],
                                     "age_before_h": h["age_before_h"], "policy": policy.name}})
        if "pass_metrics" in info:
            mp = env.map_product(env.pass_t[p])
            with rasterio.open(out / f"{date}_map.tif", "w", driver="GTiff", height=mp["flood_class"].shape[0],
                               width=mp["flood_class"].shape[1], count=3, dtype="uint16", crs=f"EPSG:{meta['epsg']}",
                               transform=tf, compress="deflate") as dst:
                for b, k in enumerate(("flood_class", "confidence", "age_h"), 1):
                    dst.write(mp[k].astype(np.uint16), b)
                    dst.set_band_description(b, k)
                dst.update_tags(date=date, policy=policy.name, event=event,
                                note="flood_class: 0 dry, 1 flood, 255 unknown; research prototype, not for operational use")
            pass_feats = [f for f in plans if f["properties"]["pass"] == p + 1]
            (out / f"{date}_plan.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": pass_feats}))
            if plan_date == date:
                return pass_feats, info["pass_metrics"]
    return None, None


def main(a):
    policy, variant = make_policy(a.policy)
    metas = {s["scene"]: s for s in event_scenes(a.event)}
    out_root = RESULTS / "products" / a.event / policy.name
    if a.plan_date:
        feats, m = run_scene(a.event, a.scene, metas[a.scene], policy, variant, out_root, a.plan_date)
        if feats is None:
            print("no acquisition on", a.plan_date, "for", a.scene)
            return
        print(f"Tasking plan {a.scene} {a.plan_date} ({policy.name}):  rank  tile  centre lon/lat  P(select)  failed")
        for f in feats:
            ring = np.array(f["geometry"]["coordinates"][0][:4])
            pr = f["properties"]
            prob = f"{pr['selection_prob']:.3f}" if pr["selection_prob"] is not None else "  -  "
            print(f"  {pr['rank']:2d}  {pr['tile']:3d}  {ring[:, 0].mean():.4f}, {ring[:, 1].mean():.4f}  {prob}  {pr['failed']}")
        print(f"map IoU vs reference after this pass: {m['iou']:.3f}  ->  {out_root / a.scene}")
        return
    names = a.scenes or sorted(metas)
    for s in names:
        run_scene(a.event, s, metas[s], policy, variant, out_root)
        print("exported", s, flush=True)
    save_json({"event": a.event, "policy": policy.name, "scenes": names, "provenance": provenance()}, out_root / "export_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="assam2016")
    ap.add_argument("--policy", default="ppo:recurrent:best")
    ap.add_argument("--scenes", nargs="*")
    ap.add_argument("--scene", default="India_00")
    ap.add_argument("--plan-date", default=None)
    main(ap.parse_args())
