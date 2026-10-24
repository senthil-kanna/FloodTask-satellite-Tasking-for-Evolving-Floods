"""Merge PPO runs trained on machine B into this repository.
  python scripts/merge_machine_b.py results_machine_b.zip
Copies results/rl/<variant>/seed<k>/ folders that are listed for machine B in configs/split.yaml and absent here.
"""
from __future__ import annotations

import json
import sys
import zipfile

from floodtask.utils import load_yaml, RESULTS, ROOT


def main(path: str):
    expected = {tuple(j.split(":")) for j in load_yaml(ROOT / "configs" / "split.yaml")["machine_b"]}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        # entries look like rl/<variant>/seed<k>/train_log.json (optionally prefixed by results/)
        runs = sorted({tuple(n.replace("\\", "/").split("rl/", 1)[1].split("/")[:2]) for n in names
                       if n.replace("\\", "/").endswith("train_log.json") and "rl/" in n.replace("\\", "/")})
        for variant, seed_dir in runs:
            seed = seed_dir.replace("seed", "")
            dest = RESULTS / "rl" / variant / seed_dir
            tag = f"{variant}:{seed}"
            if (variant, seed) not in expected:
                print(f"skip {tag}: not assigned to machine B")
                continue
            if (dest / "train_log.json").exists():
                print(f"skip {tag}: already present")
                continue
            for n in names:
                nn = n.replace("\\", "/")
                if f"rl/{variant}/{seed_dir}/" in nn and not nn.endswith("/"):
                    out = dest / nn.split(f"rl/{variant}/{seed_dir}/")[1]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(n))
            log = json.loads((dest / "train_log.json").read_text())
            print(f"merged {tag}: best val IoU {log['best_val_mean_iou']:.4f}, gpu {log['provenance'].get('gpu')}")
    missing = [f"{v}:{s}" for v, s in sorted(expected) if not (RESULTS / "rl" / v / f"seed{s}" / "train_log.json").exists()]
    print("still missing from machine B:", missing or "none")


if __name__ == "__main__":
    main(sys.argv[1])
