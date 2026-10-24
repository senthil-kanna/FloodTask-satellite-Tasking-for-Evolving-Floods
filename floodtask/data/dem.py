"""Copernicus GLO-30 DEM tiles from the AWS Open Data registry (public, no account).

Licence: Copernicus DEM GLO-30 is provided free of charge under the ESA/Copernicus licence
(see docs/dataset_inventory.md for the exact attribution string).
"""
from __future__ import annotations

import math
from pathlib import Path

import requests

from floodtask.utils import DATA, load_yaml, ROOT

RAW = DATA / "raw" / "copdem"


def tile_name(lat: int, lon: int, cfg: dict) -> str:
    return cfg["name_template"].format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))


def tiles_for_bbox(bbox) -> list[tuple[int, int]]:
    w, s, e, n = bbox
    return [(la, lo) for la in range(math.floor(s), math.floor(n) + 1) for lo in range(math.floor(w), math.floor(e) + 1)]


def fetch_tiles(bbox) -> list[Path]:
    cfg = load_yaml(ROOT / "configs" / "data.yaml")["copernicus_dem"]
    paths = []
    for la, lo in tiles_for_bbox(bbox):
        name = tile_name(la, lo, cfg)
        dest = RAW / f"{name}.tif"
        if not dest.exists():
            r = requests.get(cfg["url_template"].format(name=name), timeout=300)
            if r.status_code in (403, 404):  # ocean tile: no product exists
                continue
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(r.content)
        paths.append(dest)
    return paths
