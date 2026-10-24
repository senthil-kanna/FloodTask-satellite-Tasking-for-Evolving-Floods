"""PPO (recurrent or feed-forward) for satellite tile tasking.

Usage: python -m floodtask.rl.train_ppo --seed 0 --variant recurrent
Checkpoint selection uses validation-event scenes only (mean per-pass IoU on fixed EpisodeSpecs).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from floodtask.env.factory import VARIANTS, make_env, scene_names, val_specs
from floodtask.eval.runner import RLPolicy, run_episode, summarize
from floodtask.rl.model import ActorCritic
from floodtask.utils import load_yaml, provenance, RESULTS, ROOT, save_json, seed_everything


def evaluate(model, device, specs, variant):
    env = make_env("val", variant, seed=0)
    pol = RLPolicy(model, device)
    res = [summarize(run_episode(env, pol, s)) for s in specs]
    model.train()
    return {k: float(np.nanmean([r[k] for r in res])) for k in res[0]}


def main(seed: int, variant: str, total_steps: int | None = None, overrides: list[str] | None = None,
         tag: str | None = None, device: str | None = None):
    cfg = load_yaml(ROOT / "configs" / "ppo.yaml")
    if total_steps:
        cfg["total_steps"] = total_steps
    for kv in overrides or []:          # e.g. gamma=0.9 (hyperparameter sweeps on validation)
        k, v_ = kv.split("=")
        cfg[k] = type(cfg[k])(float(v_)) if not isinstance(cfg[k], str) else v_
    v = VARIANTS[variant]
    torch.set_num_threads(cfg.get("torch_threads", 4))   # several runs share the CPU
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = cfg["num_envs"]
    envs = [make_env("train", variant, seed=seed * 1000 + i) for i in range(n)]
    model = ActorCritic(recurrent=v["recurrent"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], eps=1e-5)
    specs = val_specs(cfg["n_val_episodes"])
    T = envs[0].P * envs[0].K
    iters = cfg["total_steps"] // (T * n)
    out = (RESULTS / "rl" / variant / f"seed{seed}") if tag is None else (RESULTS / "sweep" / tag / variant / f"seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    log, best, t_start = [], -np.inf, time.time()
    print(f"{variant} seed {seed}: {iters} iterations x {T * n} steps, params "
          f"{sum(p.numel() for p in model.parameters())}", flush=True)

    for it in range(iters):
        frac = 1 - it / iters
        opt.param_groups[0]["lr"] = cfg["lr"] * frac
        obs = [e.reset() for e in envs]
        state = model.initial_state(n, device)
        init_state = tuple(s.clone() for s in state)
        buf = {k: [] for k in ("grid", "vec", "mask", "act", "logp", "val", "rew")}
        ep_ret = np.zeros(n)
        pass_iou = []
        with torch.no_grad():
            for t in range(T):
                grid = torch.as_tensor(np.stack([o["grid"] for o in obs]), device=device)[None]
                vec = torch.as_tensor(np.stack([o["vec"] for o in obs]), device=device)[None]
                mask = torch.as_tensor(np.stack([o["mask"] for o in obs]), device=device)[None]
                logits, value, state = model(grid, vec, mask, state)
                dist = torch.distributions.Categorical(logits=logits[0])
                a = dist.sample()
                buf["grid"].append(grid[0]); buf["vec"].append(vec[0]); buf["mask"].append(mask[0])
                buf["act"].append(a); buf["logp"].append(dist.log_prob(a)); buf["val"].append(value[0])
                rews = []
                for i, e in enumerate(envs):
                    o, r, d, info = e.step(int(a[i]))
                    obs[i] = o
                    rews.append(r)
                    if "pass_metrics" in info and d:
                        pass_iou.append(info["pass_metrics"]["iou"])
                buf["rew"].append(torch.as_tensor(rews, dtype=torch.float32, device=device))
                ep_ret += rews
        B = {k: torch.stack(v_) for k, v_ in buf.items()}   # [T,n,...]
        # GAE; every rollout ends at episode termination -> bootstrap value 0
        adv = torch.zeros_like(B["rew"]); last = torch.zeros(n, device=device)
        for t in reversed(range(T)):
            nv = B["val"][t + 1] if t + 1 < T else torch.zeros(n, device=device)
            delta = B["rew"][t] + cfg["gamma"] * nv - B["val"][t]
            last = delta + cfg["gamma"] * cfg["gae_lambda"] * last
            adv[t] = last
        ret = adv + B["val"]
        env_idx = np.arange(n)
        stats = []
        for _ in range(cfg["epochs"]):
            np.random.shuffle(env_idx)
            for mb in np.array_split(env_idx, cfg["minibatches"]):
                mb_t = torch.as_tensor(mb, device=device)
                st = tuple(s[:, mb_t] for s in init_state)
                logits, value, _ = model(B["grid"][:, mb_t], B["vec"][:, mb_t], B["mask"][:, mb_t], st)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(B["act"][:, mb_t])
                a_mb = adv[:, mb_t]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                ratio = (logp - B["logp"][:, mb_t]).exp()
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * a_mb).mean()
                vl = F.mse_loss(value, ret[:, mb_t])
                # entropy over feasible actions only
                p = torch.softmax(logits, -1)
                ent = -(p * torch.log_softmax(logits, -1)).sum(-1).mean()
                loss = pg + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - ratio.log()).mean()
                stats.append([pg.item(), vl.item(), ent.item(), kl.item()])
        s = np.mean(stats, 0)
        rec = {"iter": it, "steps": (it + 1) * T * n, "return": float(ep_ret.mean()),
               "train_final_iou": float(np.mean(pass_iou)) if pass_iou else None,
               "pg_loss": s[0], "v_loss": s[1], "entropy": s[2], "approx_kl": s[3], "minutes": (time.time() - t_start) / 60}
        if (it + 1) % 5 == 0:
            print(f"  it {it+1}/{iters} {rec['minutes']:.1f} min ret {rec['return']:.3f} ent {s[2]:.2f}", flush=True)
        if (it + 1) % cfg["eval_every"] == 0 or it == iters - 1:
            ev = evaluate(model, device, specs, variant)
            rec.update({f"val_{k}": v_ for k, v_ in ev.items()})
            if ev["mean_iou"] > best:
                best = ev["mean_iou"]
                torch.save(model.state_dict(), out / "best.pt")
            torch.save(model.state_dict(), out / "last.pt")
            print(f"it {it+1}/{iters} ret {rec['return']:.3f} ent {s[2]:.2f} kl {s[3]:.4f} "
                  f"val mean IoU {ev['mean_iou']:.4f} final {ev['final_iou']:.4f} (best {best:.4f}) "
                  f"{rec['minutes']:.1f} min", flush=True)
        log.append(rec)
    save_json({"seed": seed, "variant": variant, "variant_cfg": v, "ppo": cfg, "best_val_mean_iou": best,
               "train_minutes": (time.time() - t_start) / 60,
               "n_params": sum(p.numel() for p in model.parameters()), "log": log, "provenance": provenance()},
              out / "train_log.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="recurrent", choices=list(VARIANTS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="override configs/ppo.yaml entries, key=value")
    ap.add_argument("--tag", default=None, help="write to results/sweep/<tag>/ instead of results/rl/")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    a = ap.parse_args()
    main(a.seed, a.variant, a.steps, a.set, a.tag, a.device)
