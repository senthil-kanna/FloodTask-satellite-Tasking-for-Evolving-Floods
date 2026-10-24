"""Select PPO hyperparameters from the validation sweep (results/sweep/*/feedforward/seed0/train_log.json):
highest best-validation mean IoU (flood-present passes). Writes the choice into configs/ppo.yaml and
results/ppo_sweep.json. The selection uses validation scenes only.
"""
from __future__ import annotations

import yaml

from floodtask.utils import load_json, load_yaml, provenance, RESULTS, ROOT, save_json

KEYS = ["gamma", "lr", "ent_coef", "epochs"]


def main():
    rows = []
    for log in sorted((RESULTS / "sweep").glob("*/feedforward/seed0/train_log.json")):
        L = load_json(log)
        curve = [(r["iter"] + 1, r["val_mean_iou"]) for r in L["log"] if "val_mean_iou" in r]
        rows.append({"tag": log.parts[-4], **{k: L["ppo"][k] for k in KEYS}, "best_val_mean_iou": L["best_val_mean_iou"],
                     "final_val_mean_iou": curve[-1][1], "curve": curve, "minutes": L["train_minutes"]})
    rows.sort(key=lambda r: -r["best_val_mean_iou"])
    for r in rows:
        print(f"{r['tag']:14s} gamma {r['gamma']:<5} lr {r['lr']:<7} ent {r['ent_coef']:<6} epochs {r['epochs']}  "
              f"best {r['best_val_mean_iou']:.4f} final {r['final_val_mean_iou']:.4f}")
    best = rows[0]
    cfg = load_yaml(ROOT / "configs" / "ppo.yaml")
    for k in KEYS:
        cfg[k] = best[k]
    with open(ROOT / "configs" / "ppo.yaml", "w") as f:
        f.write("# PPO settings. gamma/lr/ent_coef/epochs selected on VALIDATION by scripts/pick_sweep.py "
                f"(sweep tag '{best['tag']}', results/ppo_sweep.json)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    save_json({"chosen": best["tag"], "rows": rows, "criterion": "best validation mean IoU (flood-present passes)",
               "provenance": provenance()}, RESULTS / "ppo_sweep.json")
    print("chosen:", best["tag"])


if __name__ == "__main__":
    main()
