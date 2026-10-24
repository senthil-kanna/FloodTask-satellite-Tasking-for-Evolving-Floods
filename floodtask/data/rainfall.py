"""ERA5 daily precipitation around each Sen1Floods11 event via the Open-Meteo archive API (no key).

STATUS: RETROSPECTIVE REANALYSIS. Used only to (a) document event hydro-meteorology and (b) set the range of
synthetic storm totals in the simulator. It is never given to a policy as a forecast.

Usage: python -m floodtask.data.rainfall
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import requests

from floodtask.utils import DATA, load_json, load_yaml, ROOT, save_json


def fetch_daily(lon: float, lat: float, start: str, end: str) -> dict:
    cfg = load_yaml(ROOT / "configs" / "data.yaml")["open_meteo"]
    r = requests.get(cfg["archive_url"], params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "precipitation_sum", "timezone": "GMT", "models": "era5"}, timeout=60)
    r.raise_for_status()
    js = r.json()
    return {"lon": lon, "lat": lat, "dates": js["daily"]["time"],
            "precip_mm": [x if x is not None else float("nan") for x in js["daily"]["precipitation_sum"]],
            "grid_cell": {"latitude": js["latitude"], "longitude": js["longitude"]},
            "product": "ERA5 reanalysis via Open-Meteo archive API (models=era5) - RETROSPECTIVE, not a forecast"}


def main():
    cfg = load_yaml(ROOT / "configs" / "data.yaml")["open_meteo"]
    chips = load_json(DATA / "raw" / "sen1floods11" / "chips.json")
    out = {}
    for event in sorted({c["event"] for c in chips}):
        ev = [c for c in chips if c["event"] == event]
        lon = float(np.mean([(c["bbox"][0] + c["bbox"][2]) / 2 for c in ev]))
        lat = float(np.mean([(c["bbox"][1] + c["bbox"][3]) / 2 for c in ev]))
        d = date.fromisoformat(sorted(c["datetime"] for c in ev)[0][:10])
        start, end = d - timedelta(days=cfg["window_days_before"]), d + timedelta(days=cfg["window_days_after"])
        r = requests.get(cfg["archive_url"], params={
            "latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "daily": "precipitation_sum", "timezone": "GMT", "models": "era5"}, timeout=60)
        r.raise_for_status()
        js = r.json()
        p = np.array([x if x is not None else np.nan for x in js["daily"]["precipitation_sum"]], float)
        k = cfg["window_days_before"]
        pre = p[:k + 1]
        out[event] = {
            "lon": lon, "lat": lat, "s1_date": d.isoformat(), "dates": js["daily"]["time"], "precip_mm": p.tolist(),
            "total_10d_before_mm": float(np.nansum(pre[-10:])),
            "max_3d_before_mm": float(max(np.nansum(pre[i:i + 3]) for i in range(len(pre) - 2))),
            "grid_cell": {"latitude": js["latitude"], "longitude": js["longitude"], "elevation": js.get("elevation")},
            "product": "ERA5 reanalysis via Open-Meteo archive API (models=era5)",
        }
        print(f"{event:10s} {d} 10d-before total {out[event]['total_10d_before_mm']:.1f} mm, "
              f"max 3d {out[event]['max_3d_before_mm']:.1f} mm")
    save_json(out, DATA / "processed" / "era5_event_rainfall.json")


if __name__ == "__main__":
    main()
