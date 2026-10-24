"""Render an actual episode: per pass, the hidden flood (evaluator view), the shared map estimate, errors,
and the tiles the policy acquired. Figures come from a real run of the given policy, not illustrations.

  python scripts/visualize_episode.py --policy greedy_unc --scene India_00 --seed 20000 [--passes 0 3 6 11]
  python scripts/visualize_episode.py --policy ppo:recurrent:0 --scene India_00 --seed 20000
  python scripts/visualize_episode.py --replay assam2016 --policy greedy_unc --scene India_00
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from floodtask.env.factory import env_config, make_env, sensor_params, VARIANTS  # noqa: E402
from floodtask.env.tasking_env import EpisodeSpec, TILE  # noqa: E402
from floodtask.eval.runner import RLPolicy  # noqa: E402
from floodtask.hydro.simulator import NT  # noqa: E402
from floodtask.rl.model import ActorCritic  # noqa: E402
from floodtask.utils import RESULTS  # noqa: E402


def make_policy(spec: str):
    if spec.startswith("ppo:"):
        _, variant, seed = spec.split(":")
        m = ActorCritic(recurrent=VARIANTS[variant]["recurrent"])
        m.load_state_dict(torch.load(RESULTS / "rl" / variant / f"seed{seed}" / "best.pt", map_location="cpu", weights_only=True))
        return RLPolicy(m, "cpu", name=f"ppo_{variant}_s{seed}"), variant
    import sys
    sys.path.insert(0, str(RESULTS.parent / "scripts"))
    from run_experiments import tuned
    return tuned(spec), "recurrent"


def main(a):
    pol, variant = make_policy(a.policy)
    if a.replay:
        from floodtask.env.replay_env import ReplayEnv
        env = ReplayEnv(env_config(), sensor_params(), [a.scene], a.replay, use_forecast=VARIANTS[variant]["use_forecast"])
    else:
        env = make_env("test", variant)
    obs = env.reset(EpisodeSpec(a.scene, a.seed, a.forecast_mode))
    pol.reset(0)
    frames, done = [], False
    while not done:
        p = env.pass_idx
        act = pol.act(obs, None if pol.operational else env)
        obs, _, done, info = env.step(act)
        if "pass_metrics" in info:
            sel = [h["tile"] for h in env.history if h["pass"] == p]
            fail = [h["tile"] for h in env.history if h["pass"] == p and h["failed"]]
            frames.append((p, env._truth_eval[p].copy(), (env.est_prob >= 0.5).copy(), sel, fail, info["pass_metrics"]))
    show = [f for f in frames if f[0] in (a.passes or [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1])]
    fig, axs = plt.subplots(3, len(show), figsize=(3.2 * len(show), 9.5), squeeze=False)
    hand = env.sc["hand"]
    for j, (p, truth, est, sel, fail, m) in enumerate(show):
        axs[0, j].imshow(np.clip(hand, 0, 20), cmap="Greys_r"); axs[0, j].imshow(np.ma.masked_where(~truth, truth), cmap="Blues", vmin=0, vmax=1)
        axs[0, j].set_title(f"pass {p + 1}: hidden flood\n(evaluator only)", fontsize=8)
        axs[1, j].imshow(np.clip(hand, 0, 20), cmap="Greys_r"); axs[1, j].imshow(np.ma.masked_where(~est, est), cmap="Blues", vmin=0, vmax=1)
        for t in sel:
            r, c = divmod(t, NT)
            axs[1, j].add_patch(Rectangle((c * TILE - .5, r * TILE - .5), TILE, TILE, fill=False,
                                          ec="red" if t in fail else "orange", lw=1))
        axs[1, j].set_title(f"map estimate + acquired tiles\nIoU {m['iou']:.3f}", fontsize=8)
        err = np.zeros(truth.shape + (3,)); err[truth & ~est] = (1, 0, 0); err[~truth & est] = (1, .8, 0); err[truth & est] = (0, .4, 1)
        axs[2, j].imshow(err); axs[2, j].set_title("red: missed, yellow: false alarm,\nblue: correct flood", fontsize=8)
    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    tag = f"{'replay_' + a.replay + '_' if a.replay else ''}{pol.name}_{a.scene}_{a.seed}"
    fig.suptitle(f"{pol.name} | {a.scene} | episode seed {a.seed} | orange = acquired, red = failed acquisition", fontsize=9)
    out = RESULTS / "figures" / "episodes" / f"{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="greedy_unc")
    ap.add_argument("--scene", default="India_00")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--forecast-mode", default=None)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--passes", type=int, nargs="*")
    main(ap.parse_args())
