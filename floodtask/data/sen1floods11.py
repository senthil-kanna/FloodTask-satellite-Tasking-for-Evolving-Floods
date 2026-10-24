"""Sen1Floods11 hand-labeled subset: discovery, download, STAC metadata, manifest with checksums.

Source: Bonafilia et al. (2020), CVPR Workshops. Public GCS bucket gs://sen1floods11 (no account).
Labels (LabelHand): 1 = water, 0 = not water, -1 = nodata/cloud/unlabeled.
S1Hand: 2 bands (VV, VH) sigma0 in dB, EPSG:4326, ~10 m, NaN = nodata.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

import requests

from floodtask.utils import DATA, load_yaml, ROOT, save_json, sha256

RAW = DATA / "raw" / "sen1floods11"


def list_objects(cfg: dict, layer: str) -> list[dict]:
    prefix = f"v1.1/data/flood_events/HandLabeled/{layer}/"
    items, token = [], None
    while True:
        params = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = requests.get(cfg["list_api"], params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        items += js.get("items", [])
        token = js.get("nextPageToken")
        if not token:
            return items


def _download(url: str, dest: Path, expected_size: int) -> None:
    if dest.exists() and dest.stat().st_size == expected_size:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if tmp.stat().st_size != expected_size:
        raise IOError(f"size mismatch for {url}")
    tmp.replace(dest)


def chip_metadata(cfg: dict, chip: str) -> dict:
    url = f"{cfg['base_url']}/{cfg['catalog_prefix']}/{chip}/{chip}.json"
    js = requests.get(url, timeout=60).json()
    coords = js["geometry"]["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return {
        "chip": chip,
        "event": js["properties"]["country"],
        "datetime": js["properties"]["datetime"],
        "bbox": [min(lons), min(lats), max(lons), max(lats)],
    }


def download(workers: int = 8) -> Path:
    cfg = load_yaml(ROOT / "configs" / "data.yaml")["sen1floods11"]
    jobs = []
    for layer in cfg["layers"]:
        for it in list_objects(cfg, layer):
            name = it["name"]
            url = f"https://storage.googleapis.com/sen1floods11/{name}"
            jobs.append((url, RAW / layer / Path(name).name, int(it["size"])))
    print(f"{len(jobs)} files to fetch")
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: _download(*j), jobs))

    chips = sorted({p.name.replace("_S1Hand.tif", "") for p in (RAW / "S1Hand").glob("*.tif")})
    with cf.ThreadPoolExecutor(workers) as ex:
        meta = list(ex.map(lambda c: chip_metadata(cfg, c), chips))
    save_json(meta, RAW / "chips.json")

    manifest = {
        "source": "Sen1Floods11 v1.1 HandLabeled (gs://sen1floods11)",
        "license": "CC BY 4.0 (per dataset repository README)",
        "files": {str(p.relative_to(DATA)).replace("\\", "/"): sha256(p) for _, p, _ in jobs},
    }
    save_json(manifest, RAW / "manifest.json")
    print(f"manifest: {len(manifest['files'])} files, {len(meta)} chips")
    return RAW


def verify() -> bool:
    manifest = json.loads((RAW / "manifest.json").read_text())
    bad = [k for k, v in manifest["files"].items() if sha256(DATA / k) != v]
    print("checksum failures:", bad if bad else "none")
    return not bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    verify() if a.verify else download()
