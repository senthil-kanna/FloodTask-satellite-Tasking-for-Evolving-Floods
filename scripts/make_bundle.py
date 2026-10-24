"""Build the training bundle for a second machine: code + TRAIN/VAL scenes + frozen calibration files.
Test-event scenes are NOT included. Output: dist/floodtask_train_bundle.zip
  python scripts/make_bundle.py
"""
from __future__ import annotations

import json
import zipfile

from floodtask.env.factory import scene_names
from floodtask.utils import DATA, load_json, provenance, RESULTS, ROOT, sha256

CODE = ["floodtask", "scripts", "configs", "tests", "docs"]
FILES = ["pyproject.toml", "requirements-lock.txt", "README.md", "run_machine_b.ps1", "run_machine_b.sh"]
FROZEN = ["sim_calibration.json", "sensor_model.json", "hand_prior.json", "channel_threshold.json",
          "baseline_params.json", "protocol_freeze.json", "reward_validity_val.json"]


def main():
    out = ROOT / "dist" / "floodtask_train_bundle.zip"
    out.parent.mkdir(exist_ok=True)
    names = scene_names("train") + scene_names("val")
    meta = [s for s in load_json(DATA / "processed" / "scenes.json") if s["scene"] in set(names)]
    manifest = {}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in CODE:
            for p in (ROOT / d).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    z.write(p, p.relative_to(ROOT))
        for f in FILES:
            if (ROOT / f).exists():
                z.write(ROOT / f, f)
        for f in FROZEN:
            z.write(RESULTS / f, f"results/{f}")
            manifest[f"results/{f}"] = sha256(RESULTS / f)
        for s in names:
            p = DATA / "processed" / "scenes" / f"{s}.npz"
            z.write(p, p.relative_to(ROOT))
            manifest[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)
        z.writestr("data/processed/scenes.json", json.dumps(meta, indent=1))
        z.writestr("BUNDLE_MANIFEST.json", json.dumps({"files": manifest, "n_scenes": len(names),
                                                       "built": provenance()}, indent=1, default=str))
    print(out, round(out.stat().st_size / 1e6, 1), "MB,", len(names), "scenes")


if __name__ == "__main__":
    main()
