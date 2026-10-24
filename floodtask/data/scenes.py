"""Build study-area scenes around Sen1Floods11 chip locations.

Each scene: local UTM grid, CELL m resolution, SCENE x SCENE cells cropped from a buffered terrain window.
Real hand labels of chips falling inside the scene are reprojected onto the grid (water fraction per cell)
and used ONLY for validating the terrain/inundation model, never as simulator truth.

Usage: python -m floodtask.data.scenes
"""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform as warp_transform

from floodtask.data.dem import fetch_tiles
from floodtask.hydro.hand import terrain_products, zone_links
from floodtask.utils import DATA, load_json, load_yaml, RESULTS, ROOT, save_json

CELL = 60.0          # m
SCENE = 192          # cells (11.52 km)
BUFFER = 32          # cells each side for flow routing
TILE = 12            # cells per tile side -> 16 x 16 tiles
MIN_SEP_M = 6000.0   # minimum separation between scene centres of one event
OUT = DATA / "processed" / "scenes"
# Drainage threshold (cells of 60 m) for HAND channels. Selected on TRAINING events by
# scripts/select_channel_threshold.py (max median HAND-AUC vs hand labels); results/channel_threshold.json.
CHANNEL_CELLS_DEFAULT = 200


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def scene_centres(chips: list[dict]) -> list[tuple[float, float]]:
    """Greedy de-duplication of chip centres so scenes of one event do not overlap heavily."""
    centres = []
    for c in sorted(chips, key=lambda c: c["chip"]):
        lon, lat = (c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2
        ok = True
        for lo2, la2 in centres:
            dx = (lon - lo2) * 111320 * np.cos(np.radians(lat)); dy = (lat - la2) * 110540
            if np.hypot(dx, dy) < MIN_SEP_M:
                ok = False; break
        if ok:
            centres.append((lon, lat))
    return centres


def channel_cells() -> int:
    p = RESULTS / "channel_threshold.json"
    return int(load_json(p)["channel_cells"]) if p.exists() else CHANNEL_CELLS_DEFAULT


def build_scene(event: str, idx: int, lon: float, lat: float, chips: list[dict], out_dir=None,
                channel: int | None = None) -> dict | None:
    out_dir = out_dir or OUT
    epsg = utm_epsg(lon, lat)
    (x0,), (y0,) = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
    N = SCENE + 2 * BUFFER
    half = N * CELL / 2
    transform = from_origin(x0 - half, y0 + half, CELL, CELL)
    # geographic bbox of the window (+ margin) for DEM tile selection
    xs, ys = [x0 - half, x0 + half, x0 - half, x0 + half], [y0 - half, y0 - half, y0 + half, y0 + half]
    lons, lats = warp_transform(f"EPSG:{epsg}", "EPSG:4326", xs, ys)
    tiles = fetch_tiles((min(lons) - 0.01, min(lats) - 0.01, max(lons) + 0.01, max(lats) + 0.01))
    if not tiles:
        return None
    srcs = [rasterio.open(p) for p in tiles]
    mosaic, mtrans = merge(srcs)
    crs_src = srcs[0].crs
    for s in srcs:
        s.close()
    dem = np.full((N, N), np.nan, np.float32)
    reproject(mosaic[0], dem, src_transform=mtrans, src_crs=crs_src, dst_transform=transform,
              dst_crs=f"EPSG:{epsg}", resampling=Resampling.average, src_nodata=-32767, dst_nodata=np.nan)
    if np.isnan(dem).mean() > 0.02:
        return None  # coast / missing tiles: skip rather than invent terrain
    dem = np.where(np.isnan(dem), np.nanmin(dem), dem)
    channel = channel or channel_cells()
    terr = terrain_products(dem, channel_cells=channel)
    sl = slice(BUFFER, BUFFER + SCENE)
    crop = {k: v[sl, sl] for k, v in terr.items() if isinstance(v, np.ndarray) and v.ndim == 2}
    crop["hand"] = np.nan_to_num(crop["hand"], nan=99.0)
    zone = crop["zone"]
    zones = np.unique(zone[zone >= 0])
    lut = -np.ones(terr["zone_down"].size + 1, int); lut[zones] = np.arange(len(zones))
    zone_down = np.array([lut[terr["zone_down"][z]] if terr["zone_down"][z] >= 0 else -1 for z in zones])
    crop["zone"] = np.where(zone >= 0, lut[np.maximum(zone, 0)], -1)

    # Reproject real chip labels onto the scene grid (validation only)
    stransform = from_origin(x0 - SCENE * CELL / 2, y0 + SCENE * CELL / 2, CELL, CELL)
    water = np.zeros((SCENE, SCENE), np.float32); obs = np.zeros((SCENE, SCENE), np.float32)
    used = []
    for c in chips:
        b = c["bbox"]
        if not (min(lons) < b[0] < max(lons) and min(lats) < b[1] < max(lats)):
            continue
        with rasterio.open(DATA / "raw" / "sen1floods11" / "LabelHand" / f"{c['chip']}_LabelHand.tif") as src:
            lab = src.read(1).astype(np.float32)
            w_src, v_src = (lab == 1).astype(np.float32), (lab >= 0).astype(np.float32)
            for arr, dst in ((w_src, water), (v_src, obs)):
                tmp = np.full((SCENE, SCENE), np.nan, np.float32)
                reproject(arr, tmp, src_transform=src.transform, src_crs=src.crs, dst_transform=stransform,
                          dst_crs=f"EPSG:{epsg}", resampling=Resampling.average, dst_nodata=np.nan)
                m = np.isfinite(tmp)
                dst[m] = tmp[m] if dst is obs else np.maximum(dst[m], tmp[m])
        used.append(c["chip"])
    name = f"{event}_{idx:02d}"
    np.savez_compressed(out_dir / f"{name}.npz", dem=crop["dem"], hand=crop["hand"], acc=crop["acc"],
                        drainage=crop["drainage"], zone=crop["zone"], zone_down=zone_down,
                        down=_crop_down(terr["down"], N, sl), label_water=water, label_valid=obs)
    return {"scene": name, "event": event, "lon": lon, "lat": lat, "epsg": epsg,
            "transform": list(stransform)[:6], "chips": used, "n_zones": int(len(zones)),
            "dem_tiles": [p.name for p in tiles], "channel_cells": channel}


def _crop_down(down: np.ndarray, N: int, sl: slice) -> np.ndarray:
    """Re-index D8 downstream pointers to the cropped grid; flow leaving the crop -> -1."""
    r, c = np.divmod(np.where(down >= 0, down, 0), N)
    r, c = r.reshape(N, N)[sl, sl] - sl.start, c.reshape(N, N)[sl, sl] - sl.start
    d = down.reshape(N, N)[sl, sl]
    inside = (d >= 0) & (r >= 0) & (r < SCENE) & (c >= 0) & (c < SCENE)
    return np.where(inside, r * SCENE + c, -1).astype(np.int32)


def relink():
    """Recompute acyclic outlet-based zone links (hand.zone_links) for already-built scenes, in place.
    Needed for scenes built before the outlet rule was introduced; terrain arrays are unchanged."""
    for p in sorted(OUT.glob("*.npz")):
        d = dict(np.load(p))
        d["zone_down"] = zone_links(d["zone"], d["down"], d["drainage"], d["acc"])
        np.savez_compressed(p, **d)
    print("relinked", len(list(OUT.glob("*.npz"))), "scenes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = load_yaml(ROOT / "configs" / "data.yaml")["split"]
    ev_split = {e: s for s, evs in split.items() for e in evs}
    chips = load_json(DATA / "raw" / "sen1floods11" / "chips.json")
    meta = []
    for event in sorted({c["event"] for c in chips}):
        ev_chips = [c for c in chips if c["event"] == event]
        for i, (lon, lat) in enumerate(scene_centres(ev_chips)):
            m = build_scene(event, i, lon, lat, ev_chips)
            if m:
                m["split"] = ev_split[event]
                meta.append(m)
                print(m["scene"], m["n_zones"], "zones", len(m["chips"]), "chips", flush=True)
            else:
                print(f"skip {event} {i} (DEM missing/coastal)")
    save_json(meta, DATA / "processed" / "scenes.json")


if __name__ == "__main__":
    import sys
    relink() if "--relink" in sys.argv else main()
