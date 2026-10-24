"""Common evaluation loop: every policy runs on identical EpisodeSpecs (scene, exogenous seed, forecast mode),
so physics, swaths, failures, sensor noise and forecast errors are paired across policies."""
from __future__ import annotations

import time

import numpy as np
import torch

from floodtask.env.tasking_env import EpisodeSpec, FloodTaskingEnv
from floodtask.policies.baselines import Policy


class RLPolicy(Policy):
    def __init__(self, model, device="cpu", deterministic=True, name="ppo"):
        self.model, self.device, self.deterministic, self.name = model.eval(), device, deterministic, name

    def reset(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.state = self.model.initial_state(1, self.device)

    @torch.no_grad()
    def act(self, obs, env=None):
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)[None, None]
        logits, _, self.state = self.model(t(obs["grid"], torch.float32), t(obs["vec"], torch.float32),
                                           t(obs["mask"], torch.bool), self.state)
        logits = logits[0, 0].cpu()
        if self.deterministic:
            return int(torch.argmax(logits))
        return int(torch.multinomial(torch.softmax(logits, 0), 1, generator=self.gen))


def run_episode(env: FloodTaskingEnv, policy: Policy, spec: EpisodeSpec, policy_seed: int = 0) -> dict:
    obs = env.reset(spec)
    policy.reset(policy_seed)
    total, passes, lat = 0.0, [], []
    done = False
    while not done:
        t0 = time.perf_counter()
        a = policy.act(obs, env if not policy.operational else None)
        lat.append(time.perf_counter() - t0)
        if not obs["mask"][a]:
            raise RuntimeError(f"{policy.name} chose infeasible tile {a}")
        obs, r, done, info = env.step(a)
        total += r
        if "pass_metrics" in info:
            passes.append(info["pass_metrics"])
    return {"scene": spec.scene, "seed": spec.seed, "forecast_mode": env.fmode, "policy": policy.name,
            "return": total, "passes": passes, "decision_latency_ms_mean": 1000 * float(np.mean(lat)),
            "decision_latency_ms_p95": 1000 * float(np.percentile(lat, 95)), "history": env.history}


def summarize(ep: dict) -> dict:
    """Primary endpoint: mean IoU over flood-present passes (NaN if the episode never floods)."""
    ious = [p["iou"] for p in ep["passes"] if p.get("flood_present", True)]
    return {"final_iou": ep["passes"][-1]["iou"] if ep["passes"][-1].get("flood_present", True) else float("nan"),
            "mean_iou": float(np.mean(ious)) if ious else float("nan"), "n_flood_passes": len(ious),
            "mean_false_alarm_frac": float(np.mean([p.get("false_alarm_frac", np.nan) for p in ep["passes"]])),
            "return": ep["return"]}
