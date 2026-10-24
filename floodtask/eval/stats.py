"""Paired, cluster-aware statistics for policy comparisons (protocol fixed in docs/experimental_protocol.md).

Unit of analysis: SCENE. Episodes of one scene share terrain and are correlated; scenes of one event share the
flood regime. Pixels are never treated as independent samples.
  1. Per-episode metric for each policy (RL policies: averaged over training seeds per episode).
  2. Paired difference per episode, averaged within scene -> scene-level differences.
  3. 95% CI: bootstrap over scenes stratified by event (B = 10,000).
  4. Wilcoxon signed-rank test on scene-level differences; Holm correction across the predeclared family.
  5. Effect size: Cohen's d_z on scene-level differences; per-event mean differences also reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def scene_level(df: pd.DataFrame, metric: str, a: str, b: str) -> pd.DataFrame:
    """df columns: policy, train_seed, scene, event, episode_seed, <metric>."""
    ep = df.groupby(["policy", "scene", "event", "episode_seed"], as_index=False)[metric].mean()  # average seeds
    wa = ep[ep.policy == a].set_index(["scene", "event", "episode_seed"])[metric]
    wb = ep[ep.policy == b].set_index(["scene", "event", "episode_seed"])[metric]
    if set(wa.index) != set(wb.index):
        raise ValueError(f"unpaired episodes between {a} and {b}")
    # NaN (metric undefined, e.g. no flood at the final pass) is a property of the episode, shared by both policies
    d = (wa - wb).dropna().rename("diff").reset_index()
    return d.groupby(["scene", "event"], as_index=False)["diff"].mean()


def stratified_bootstrap_ci(sc: pd.DataFrame, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    groups = [g["diff"].to_numpy() for _, g in sc.groupby("event")]
    w = np.array([len(g) for g in groups], float); w /= w.sum()
    boots = np.zeros(n_boot)
    for i in range(n_boot):
        boots[i] = sum(wi * rng.choice(g, g.size).mean() for wi, g in zip(w, groups))
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def compare(df: pd.DataFrame, metric: str, a: str, b: str) -> dict:
    sc = scene_level(df, metric, a, b)
    d = sc["diff"].to_numpy()
    lo, hi = stratified_bootstrap_ci(sc)
    p = float(sps.wilcoxon(d).pvalue) if np.any(d != 0) and d.size >= 6 else float("nan")
    seed_rows = []
    if "train_seed" in df and df[df.policy == a]["train_seed"].nunique() > 1:
        for s, g in df[df.policy == a].groupby("train_seed"):
            sub = pd.concat([g, df[df.policy == b]])
            seed_rows.append(float(scene_level(sub, metric, a, b)["diff"].mean()))
    return {
        "a": a, "b": b, "metric": metric, "n_scenes": int(d.size), "n_events": int(sc.event.nunique()),
        "n_episodes": int(df[df.policy == a].groupby(["scene", "episode_seed"]).ngroups),
        "mean_diff": float(d.mean()), "ci95": [lo, hi], "wilcoxon_p": p,
        "cohen_dz": float(d.mean() / d.std(ddof=1)) if d.size > 1 and d.std(ddof=1) > 0 else float("nan"),
        "per_event_mean_diff": sc.groupby("event")["diff"].mean().to_dict(),
        "per_seed_mean_diff": seed_rows,
        "n_seeds_positive": int(np.sum(np.array(seed_rows) > 0)) if seed_rows else None,
    }


def holm(results: list[dict], key: str = "wilcoxon_p") -> list[dict]:
    ps = np.array([r[key] for r in results], float)
    order = np.argsort(np.where(np.isnan(ps), np.inf, ps))
    m = np.sum(~np.isnan(ps))
    adj, running = np.full(len(ps), np.nan), 0.0
    for rank, i in enumerate(order):
        if np.isnan(ps[i]):
            continue
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    for r, a in zip(results, adj):
        r["p_holm"] = float(a)
    return results


def mixed_effects(df: pd.DataFrame, metric: str, a: str, b: str) -> dict:
    """Cross-check of the bootstrap: linear mixed model on EPISODE-level paired differences,
    diff ~ 1 + (1 | scene), event as fixed effect when >1 event. Reports the intercept (mean effect), its 95% CI
    and p-value. Secondary analysis; the confirmatory test stays the scene-level Wilcoxon + Holm."""
    import statsmodels.formula.api as smf
    ep = df.groupby(["policy", "scene", "event", "episode_seed"], as_index=False)[metric].mean()
    wa = ep[ep.policy == a].set_index(["scene", "event", "episode_seed"])[metric]
    wb = ep[ep.policy == b].set_index(["scene", "event", "episode_seed"])[metric]
    d = (wa - wb).dropna().rename("diff").reset_index()
    formula = "diff ~ 1" if d.event.nunique() == 1 else "diff ~ C(event, Sum)"
    try:
        fit = smf.mixedlm(formula, d, groups=d["scene"]).fit(reml=True)
        lo, hi = fit.conf_int().loc["Intercept"]
        return {"a": a, "b": b, "metric": metric, "intercept": float(fit.params["Intercept"]), "ci95": [float(lo), float(hi)],
                "p": float(fit.pvalues["Intercept"]), "scene_var": float(fit.cov_re.iloc[0, 0]), "resid_var": float(fit.scale),
                "n_episodes": int(len(d)), "n_scenes": int(d.scene.nunique()), "converged": bool(fit.converged)}
    except Exception as e:  # report, never hide, a failed fit
        return {"a": a, "b": b, "metric": metric, "error": repr(e)}
