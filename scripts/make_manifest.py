"""Checksum manifest of all derived data, configs, checkpoints and raw results (reproducibility package).
  python scripts/make_manifest.py            # write results/data_manifest.json
  python scripts/make_manifest.py --verify   # verify current files against it
"""
from __future__ import annotations

import argparse

from floodtask.utils import load_json, provenance, RESULTS, ROOT, save_json, sha256

GLOBS = ["data/raw/copdem/*.tif", "data/processed/*.json", "data/processed/scenes/*.npz",
         "data/processed/real_event/**/*", "configs/*.yaml", "docs/experimental_protocol.md",
         "results/segmentation/**/*.pt", "results/rl/**/*.pt", "results/*.json", "results/experiments/**/*"]


def files():
    out = []
    for g in GLOBS:
        out += [p for p in ROOT.glob(g) if p.is_file()]
    return sorted(set(out))


def main(verify: bool):
    path = RESULTS / "data_manifest.json"
    if verify:
        ref = load_json(path)["files"]
        bad = [k for k, v in ref.items() if not (ROOT / k).exists() or sha256(ROOT / k) != v]
        print(f"{len(ref)} files checked, {len(bad)} mismatches", *bad[:20], sep="\n")
        return
    m = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in files() if p != path}
    save_json({"files": m, "provenance": provenance()}, path)
    print(len(m), "files hashed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    main(ap.parse_args().verify)
