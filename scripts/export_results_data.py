"""Export the results tables (leaderboard, paired tests, per-pass curves, robustness, Assam, training runs) as JSON
row files for the Results tabs of the viewer (results/viewer/index.html).

  python scripts/export_results_data.py            # writes results/viewer/data/results/
Every value is read from results/analysis, results/experiments and results/rl; nothing is recomputed differently
from scripts/analyze.py (its loader is reused)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import yaml  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "viewer" / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)
FAMILY = lambda p: "oracle" if p == "oracle" else ("rl" if p.startswith("ppo") else "baseline")


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def dump(name, rows):
    rows = [{k: clean(v) for k, v in r.items()} for r in rows]
    (OUT / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    print(name, len(rows), (OUT / f"{name}.json").stat().st_size)


AN = ROOT / "results" / "analysis"
# leaderboard
rows = []
for scope in ("all_test_events", "India"):
    t = pd.read_csv(AN / f"core_table_{scope}.csv")
    for _, r in t.iterrows():
        rows.append({"scope": scope, "policy": r.policy, "label": A.NAMES.get(r.policy, r.policy), "family": FAMILY(r.policy),
                     "mean_iou": r.mean_iou_mean, "mean_iou_std": r.mean_iou_std, "final_iou": r.final_iou_mean,
                     "final_f1": r.final_f1_mean, "precision": r.final_precision_mean, "recall": r.final_recall_mean,
                     "boundary_f1": r.final_boundary_f1_mean, "area_bias": r.final_area_bias_mean,
                     "time_to_thresh": r.time_to_thresh_mean, "reached": r.reached_mean,
                     "false_alarm": r.false_alarm_frac_mean, "latency_ms": r.latency_ms_mean, "n_episodes": r.n_episodes})
dump("leaderboard", rows)

# paired tests
rows = []
for scope in ("all_test_events", "India"):
    s = json.load(open(AN / f"stats_core_{scope}.json"))
    me = {m["b"]: m for m in s["mixed_effects_crosscheck"]}
    for r in s["confirmatory"]:
        m = me.get(r["b"], {})
        ev = r["per_event_mean_diff"]
        rows.append({"scope": scope, "comparator": r["b"], "label": A.NAMES.get(r["b"], r["b"]), "family": FAMILY(r["b"]),
                     "diff": r["mean_diff"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1], "wilcoxon_p": r["wilcoxon_p"],
                     "p_holm": r["p_holm"], "cohen_dz": r["cohen_dz"], "seeds_positive": r["n_seeds_positive"],
                     "n_seeds": len(r["per_seed_mean_diff"]), "n_scenes": r["n_scenes"],
                     "bolivia": ev.get("Bolivia"), "india": ev.get("India"), "mekong": ev.get("Mekong"),
                     "me_diff": m.get("intercept"), "me_lo": (m.get("ci95") or [None])[0], "me_hi": (m.get("ci95") or [None, None])[1],
                     "me_p": m.get("p")})
dump("paired", rows)

# load experiment tables
EXP = ROOT / "results" / "experiments"
data = {p.name: A.load(p.name) for p in EXP.iterdir() if (p / "episodes.jsonl").exists()}
core_all = pd.concat([data["core"], data["ablation"]])


def curve_rows(df, setname):
    cols = [c for c in df.columns if c.startswith("iou_p")]
    out = []
    for pol, g in df.groupby("policy"):
        m = g.groupby(["scene", "episode_seed"])[cols].mean().mean(skipna=True)
        for i, v in enumerate(m.values):
            out.append({"set": setname, "policy": pol, "label": A.NAMES.get(pol, pol), "family": FAMILY(pol), "pass": i + 1, "iou": v})
    return out


rows = curve_rows(core_all, "all_test_events") + curve_rows(core_all[core_all.event == "India"], "India")
for ev in ("real_assam2016", "real_assam2022"):
    d = data[ev]
    for mode in ("operational", "missing"):
        rows += curve_rows(d[d.forecast_mode == mode], f"{ev.replace('real_', '')}_{mode}")
dump("per_pass", rows)

# real events summary
rows = []
for ev in ("real_assam2016", "real_assam2022"):
    d = data[ev]
    tab = d.groupby(["forecast_mode", "policy"]).agg(mean_iou=("mean_iou", "mean"), final_iou=("final_iou", "mean"),
                                                     time_to_thresh=("time_to_thresh", "mean"), reached=("reached", "mean"),
                                                     false_alarm=("false_alarm_frac", "mean"), n=("mean_iou", "size")).reset_index()
    for _, r in tab.iterrows():
        rows.append({"event": ev.replace("real_", ""), "mode": r.forecast_mode, "key": f"{ev.replace('real_', '')}|{r.forecast_mode}|{r.policy}",
                     "policy": r.policy, "label": A.NAMES.get(r.policy, r.policy), "family": FAMILY(r.policy),
                     "mean_iou": r.mean_iou, "final_iou": r.final_iou, "time_to_thresh": r.time_to_thresh, "reached": r.reached,
                     "false_alarm": r.false_alarm, "n_episodes": r.n, "n_passes": len([c for c in d.columns if c.startswith("iou_p")])})
dump("real_events", rows)

# label check
m = json.load(open(EXP / "real_assam2016" / "meta.json"))["label_check"]
dump("label_check", [{"product": "Reference map", "key": "reference", **{k: m["reference"][k] for k in ("iou", "precision", "recall", "f1")}, "date": m["label_date"], "cells": m["n_labelled_cells_60m"]},
                     {"product": "Deployed U-Net", "key": "unet_obs", **{k: m["unet_obs"][k] for k in ("iou", "precision", "recall", "f1")}, "date": m["label_date"], "cells": m["n_labelled_cells_60m"]}])

# conditions (long)
c = pd.read_csv(AN / "conditions_mean_iou.csv", index_col=0)
core_key = [i for i in c.index if i.startswith("core")][0]
GROUP = lambda k: "Core" if k.startswith("core") else k.split("_")[0].capitalize()
NICE = {"budget_8": "8 tiles per pass", "budget_32": "32 tiles per pass", "revisit_6h": "6 h revisit", "revisit_24h": "24 h revisit",
        "latency_1": "1-pass delivery delay", "sensor_perfect": "Perfect sensor", "sensor_degraded": "Degraded sensor",
        core_key: "Core (16 tiles, 12 h, no delay)"}
rows = []
for k, r in c.iterrows():
    for pol, v in r.items():
        if pd.isna(v):
            continue
        base = c.loc[core_key, pol]
        rows.append({"condition": k, "key": f"{k}|{pol}", "condition_label": NICE.get(k, k.replace("forecast_", "Forecast: ").replace("_", " ")),
                     "group": GROUP(k), "policy": pol, "label": A.NAMES.get(pol, pol), "family": FAMILY(pol),
                     "iou": v, "delta_vs_core": None if pd.isna(base) else v - base})
dump("conditions", rows)

# strata
s = pd.read_csv(AN / "flood_size_strata.csv")
dump("strata", [{"key": f"{r.policy}|{k}", "policy": r.policy, "label": A.NAMES.get(r.policy, r.policy), "family": FAMILY(r.policy),
                 "stratum": k, "iou": r[k]} for _, r in s.iterrows() for k in ("small", "medium", "large")])

# runs + curves
split = yaml.safe_load(open(ROOT / "configs" / "split.yaml"))
mach = {s: "A" for s in split["machine_a"]} | {s: "B" for s in split["machine_b"]}
runs, curves = [], []
for f in sorted((ROOT / "results" / "rl").glob("*/seed*/train_log.json")):
    t = json.load(open(f)); var, seed = t["variant"], t["seed"]
    ev = [e for e in t["log"] if "val_mean_iou" in e]
    best = max(ev, key=lambda e: e["val_mean_iou"])
    runs.append({"run": f"{var}:{seed}", "variant": var, "policy": "ppo_" + var, "label": A.NAMES.get("ppo_" + var, var), "seed": seed,
                 "best_val_iou": t["best_val_mean_iou"], "best_steps": best["steps"], "last_val_iou": ev[-1]["val_mean_iou"],
                 "best_val_false_alarm": best["val_mean_false_alarm_frac"], "train_minutes": t["train_minutes"],
                 "n_params": t["n_params"], "steps": t["log"][-1]["steps"], "machine": mach.get(f"{var}:{seed}", "?"),
                 "gpu": t["provenance"].get("gpu")})
    for e in ev:
        curves.append({"run": f"{var}:{seed}", "variant": var, "seed": seed, "steps": e["steps"], "val_iou": e["val_mean_iou"],
                       "val_false_alarm": e["val_mean_false_alarm_frac"], "train_return": e["return"]})
dump("runs", runs)
dump("val_curves", curves)
