"""Multi-date real-event data: Sentinel-1 RTC time series over held-out test scenes (configs/data.yaml: real_events).
Default target: Assam (India), Brahmaputra valley - 2016 and 2022 monsoon floods.

Source: Microsoft Planetary Computer `sentinel-1-rtc` (terrain-corrected gamma0, linear, 10 m, UTM),
anonymous signed access. Sen1Floods11 S1Hand is GEE GRD sigma0 (ellipsoid), so we apply a FIXED,
a-priori conversion sigma0 ~= gamma0 * cos(39 deg) (mid-swath IW incidence) -- not tuned on any label.

Steps (python -m floodtask.data.real_event --event assam2016 --step download|predict|rain|check):
  download: per scene and acquisition date, mosaic VV/VH onto the scene grid at 10 m -> data/processed/real_event/
  predict:  U-Net observations (seed 0, the deployed model), reference masks (U-Net seeds 1+2 ensemble AND
            per-date VH Otsu threshold must agree; disagreement -> excluded), all aggregated to the 60 m grid.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import torch
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform_bounds

from floodtask.data.scenes import CELL, SCENE
from floodtask.utils import DATA, load_json, load_yaml, provenance, RESULTS, ROOT, save_json

ROOT_OUT = DATA / "processed" / "real_event"
FINE = 10.0
NF = int(SCENE * CELL / FINE)        # 1152 fine pixels per side
COS_INC = float(np.cos(np.radians(39.0)))


def event_cfg(key: str) -> dict:
    return load_yaml(ROOT / "configs" / "data.yaml")["real_events"][key]


def scenes(key: str):
    """Non-overlapping subset of the event's scenes (greedy in name order, centres >= one scene width apart).
    Chosen geometrically, independent of any result, to bound download volume and avoid double-counting area."""
    ev = event_cfg(key)["scene_event"]
    out = []
    for s in sorted((s for s in load_json(DATA / "processed" / "scenes.json") if s["event"] == ev), key=lambda s: s["scene"]):
        x, y = s["transform"][2], s["transform"][5]
        if all(s["epsg"] != o["epsg"] or max(abs(x - o["transform"][2]), abs(y - o["transform"][5])) >= SCENE * CELL
               for o in out):
            out.append(s)
    return out


def download(key: str):
    WINDOW = event_cfg(key)["window"]
    OUT = ROOT_OUT / key
    OUT.mkdir(parents=True, exist_ok=True)
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1/",
                                    modifier=planetary_computer.sign_inplace)
    meta = load_json(OUT / "acquisitions.json")["scenes"] if (OUT / "acquisitions.json").exists() else []
    done = {m["scene"] for m in meta}
    for s in scenes(key):
        if s["scene"] in done and (OUT / f"{s['scene']}_s1.npz").exists():
            continue   # resumable
        t = s["transform"]
        tf = Affine(FINE, 0, t[2], 0, -FINE, t[5])
        crs = f"EPSG:{s['epsg']}"
        bbox = transform_bounds(crs, "EPSG:4326", t[2], t[5] - SCENE * CELL, t[2] + SCENE * CELL, t[5])
        items = cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=WINDOW).item_collection()
        by_date, skipped = defaultdict(list), []
        for it in items:
            if "vv" in it.assets and "vh" in it.assets:   # U-Net needs dual-pol VV+VH
                by_date[it.datetime.date().isoformat()].append(it)
            else:
                skipped.append({"item": it.id, "date": it.datetime.date().isoformat(), "assets": sorted(it.assets)})
        dates, stack, info = [], [], []
        for d in sorted(by_date):
            bands = []
            for pol in ("vv", "vh"):
                arr = np.full((NF, NF), np.nan, np.float32)
                for it in by_date[d]:
                    tmp = _read_warped(it.assets[pol].href, tf, crs)
                    fill = np.isnan(arr) & np.isfinite(tmp) & (tmp > 0)
                    arr[fill] = tmp[fill]
                bands.append(10 * np.log10(np.maximum(arr * COS_INC, 1e-6)))
            valid = np.isfinite(bands[0]).mean()
            if valid < 0.2:
                continue
            dates.append(d); stack.append(np.stack(bands).astype(np.float16))
            info.append({"date": d, "items": [it.id for it in by_date[d]], "valid_frac": float(valid),
                         "orbit": by_date[d][0].properties.get("sat:orbit_state"),
                         "rel_orbit": by_date[d][0].properties.get("sat:relative_orbit")})
            print(s["scene"], d, f"valid {valid:.2f}", flush=True)
        np.savez_compressed(OUT / f"{s['scene']}_s1.npz", dates=np.array(dates), s1=np.stack(stack))
        meta.append({"scene": s["scene"], "acquisitions": info, "skipped_not_dual_pol": skipped})
        _save_meta(key, OUT, WINDOW, meta)
    _save_meta(key, OUT, WINDOW, meta)


def _save_meta(key, OUT, WINDOW, meta):
    save_json({"event": key, "window": WINDOW, "collection": "sentinel-1-rtc (Microsoft Planetary Computer)",
               "sigma0_conversion": f"gamma0 * cos(39 deg) = {COS_INC:.4f}", "scenes": meta,
               "provenance": provenance()}, OUT / "acquisitions.json")


def _read_warped(href, tf, crs, tries=4):
    """Re-sign (SAS tokens expire ~1 h) and warp onto the scene grid, retrying transient read errors."""
    import time
    for k in range(tries):
        try:
            tmp = np.full((NF, NF), np.nan, np.float32)
            with rasterio.open(planetary_computer.sign(href.split("?")[0])) as src:
                reproject(rasterio.band(src, 1), tmp, dst_transform=tf, dst_crs=crs,
                          resampling=Resampling.average, src_nodata=-32768, dst_nodata=np.nan)
            return tmp
        except rasterio.errors.RasterioError:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def otsu(x):
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, 256)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1); m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def block(a, b=6):
    return a.reshape(SCENE, b, SCENE, b).mean((1, 3))


def predict(key: str, only: list[str] | None = None, device: str | None = None):
    OUT = ROOT_OUT / key
    from floodtask.seg.dataset import normalize
    from floodtask.seg.evaluate import load_model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    stats = load_json(DATA / "processed" / "s1_norm_stats.json")
    models = {k: load_model(k, device) for k in (0, 1, 2)}
    prev = RESULTS / f"real_event_{key}_references.json"
    summary = load_json(prev)["per_scene_date"] if prev.exists() else []
    for s in scenes(key):
        if only and s["scene"] not in only:
            continue
        if (OUT / f"{s['scene']}_masks.npz").exists() and any(x["scene"] == s["scene"] for x in summary):
            continue   # resumable
        z = np.load(OUT / f"{s['scene']}_s1.npz")
        obs_p, obs_v, ref, ref_v = [], [], [], []
        for d, s1 in zip(z["dates"], z["s1"].astype(np.float32)):
            x, valid = normalize(s1, stats)
            xt = torch.from_numpy(x)[None].to(device)
            with torch.no_grad():
                p = {k: m.predict_proba(xt)[0].cpu().numpy() for k, m in models.items()}
            thr = otsu(s1[1][valid])
            o = (s1[1] < thr) & valid
            ens = (p[1] + p[2]) / 2 >= 0.5
            agree = valid & (ens == o)
            vf = block(valid.astype(float))
            obs_p.append(block(np.where(valid, p[0], 0)) / np.maximum(vf, 1e-9)); obs_v.append(vf >= 0.5)
            af = block(agree.astype(float))
            ref.append(block((ens & agree).astype(float)) / np.maximum(af, 1e-9) >= 0.5); ref_v.append(af >= 0.8)
            summary.append({"scene": s["scene"], "date": str(d), "otsu_vh_db": float(thr), "valid_frac": float(valid.mean()),
                            "ref_agreement_frac": float(agree.sum() / max(valid.sum(), 1)),
                            "ref_water_frac": float(ref[-1][ref_v[-1]].mean()) if ref_v[-1].any() else None})
        np.savez_compressed(OUT / f"{s['scene']}_masks.npz", dates=z["dates"], obs_prob=np.stack(obs_p).astype(np.float32),
                            obs_valid=np.stack(obs_v), ref=np.stack(ref), ref_valid=np.stack(ref_v))
        print(s["scene"], len(z["dates"]), "dates", flush=True)
        save_json({"per_scene_date": summary, "provenance": provenance()}, prev)
    save_json({"per_scene_date": summary, "provenance": provenance()}, RESULTS / f"real_event_{key}_references.json")


def rain(key: str):
    """ERA5 daily precipitation (RETROSPECTIVE) at the replay-area centroid for the replay window."""
    from floodtask.data.rainfall import fetch_daily
    sc = scenes(key)
    lon, lat = float(np.mean([s["lon"] for s in sc])), float(np.mean([s["lat"] for s in sc]))
    start, end = event_cfg(key)["window"].split("/")
    out = fetch_daily(lon, lat, start, end)
    save_json(out, ROOT_OUT / key / "era5_daily.json")
    print(key, "ERA5 window total", round(float(np.nansum(out["precip_mm"])), 1), "mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="assam2016")
    ap.add_argument("--step", choices=["download", "predict", "rain"], required=True)
    ap.add_argument("--scenes", nargs="*", help="predict: restrict to these scenes")
    ap.add_argument("--device", default=None, help="predict: cuda | cpu")
    a = ap.parse_args()
    if a.step == "predict":
        predict(a.event, a.scenes, a.device)
    else:
        {"download": download, "rain": rain}[a.step](a.event)
