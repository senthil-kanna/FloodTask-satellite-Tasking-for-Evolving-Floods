import numpy as np
import pandas as pd

from floodtask.eval.stats import compare, holm


def frame(effect):
    rng = np.random.default_rng(0)
    rows = []
    for e in range(3):
        for s in range(5):
            base = rng.normal(0.5, 0.1)
            for ep in range(4):
                noise = rng.normal(0, 0.02)
                rows.append(dict(policy="A", train_seed=0, scene=f"{e}_{s}", event=str(e), episode_seed=ep, iou=base + effect + noise))
                rows.append(dict(policy="B", train_seed=-1, scene=f"{e}_{s}", event=str(e), episode_seed=ep, iou=base + noise))
    return pd.DataFrame(rows)


def test_compare_detects_effect_and_null():
    r = compare(frame(0.05), "iou", "A", "B")
    assert r["n_scenes"] == 15 and abs(r["mean_diff"] - 0.05) < 1e-9
    assert r["ci95"][0] > 0 and r["wilcoxon_p"] < 0.01
    r0 = compare(frame(0.0), "iou", "A", "B")
    assert r0["ci95"][0] <= 0 <= r0["ci95"][1]


def test_holm_monotone():
    res = holm([{"wilcoxon_p": p} for p in (0.01, 0.04, 0.03)])
    adj = [r["p_holm"] for r in res]
    assert np.isclose(adj[0], 0.03) and np.isclose(adj[2], 0.06) and np.isclose(adj[1], 0.06)


def test_mixed_effects_recovers_effect():
    from floodtask.eval.stats import mixed_effects
    r = mixed_effects(frame(0.05), "iou", "A", "B")
    assert "error" not in r and abs(r["intercept"] - 0.05) < 0.01 and r["ci95"][0] > 0
