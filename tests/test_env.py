import numpy as np
import pytest
import torch

import floodtask.hydro.simulator as sim
from floodtask.data.scenes import SCENE, _crop_down
from floodtask.env.factory import env_config
from floodtask.env.sensor import PERFECT, SensorParams
from floodtask.env.tasking_env import GRID_CHANNELS, IOU_EPS, N_TILES, TILE, VEC_FEATURES, EpisodeSpec, FloodTaskingEnv, tile_view
from floodtask.eval.runner import RLPolicy, run_episode
from floodtask.hydro.hand import terrain_products
from floodtask.policies.baselines import BASELINES
from floodtask.rl.model import ActorCritic


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scenes")
    N, B = SCENE + 64, 32
    r, c = np.indices((N, N))
    dem = (np.abs(c - N // 2) * 0.05 + (N - r) * 0.02 + 0.3 * np.sin(r / 7.0) * np.cos(c / 9.0)).astype(np.float32)
    t = terrain_products(dem, channel_cells=200, zone_block=48)
    sl = slice(B, B + SCENE)
    zone = t["zone"][sl, sl]
    zones = np.unique(zone[zone >= 0])
    lut = -np.ones(t["zone_down"].size + 1, int); lut[zones] = np.arange(len(zones))
    zd = np.array([lut[t["zone_down"][z]] if t["zone_down"][z] >= 0 else -1 for z in zones])
    np.savez(d / "valley_00.npz", dem=t["dem"][sl, sl], hand=np.nan_to_num(t["hand"][sl, sl], nan=99.0),
             acc=t["acc"][sl, sl], drainage=t["drainage"][sl, sl], zone=np.where(zone >= 0, lut[np.maximum(zone, 0)], -1),
             zone_down=zd, down=_crop_down(t["down"], N, sl), label_water=np.zeros((SCENE, SCENE), np.float32),
             label_valid=np.zeros((SCENE, SCENE), np.float32))
    sim.SCENE_DIR = d
    sim.load_scene.cache_clear()
    return "valley_00"


def make(scene, sensor=SensorParams(), **over):
    cfg = env_config({"physics": {"storm_total_mm": [150, 250], "ext_scale": [2, 4], "runoff_coef": [0.3, 0.7],
                                  "loss_mm": [0, 0]}, **over})
    return FloodTaskingEnv(cfg, sensor, [scene], seed=0)


def test_shapes_and_episode_length(scene):
    env = make(scene)
    obs = env.reset(EpisodeSpec(scene, 1))
    assert obs["grid"].shape == (len(GRID_CHANNELS), 16, 16) and obs["vec"].shape == (len(VEC_FEATURES),)
    assert obs["mask"].shape == (N_TILES,) and obs["grid"].dtype == np.float32
    assert env._truth_eval.any(), "test scene should flood"
    pol = BASELINES["random"]()
    ep = run_episode(env, pol, EpisodeSpec(scene, 1))
    assert len(ep["history"]) == env.P * env.K and len(ep["passes"]) == env.P


def test_no_future_truth_leakage(scene):
    env = make(scene)
    env.reset(EpisodeSpec(scene, 2))
    for _ in range(env.K):  # advance one pass
        env.step(int(np.flatnonzero(env.observation()["mask"])[0]))
    o1 = env.observation()
    t = int(env.pass_t[env.pass_idx])
    env._truth_eval[:] = ~env._truth_eval
    env._truth_acq[env.pass_idx:] = ~env._truth_acq[env.pass_idx:]
    env._level[t + 1:] += 100.0
    o2 = env.observation()
    for k in o1:
        assert np.array_equal(o1[k], o2[k]), k


def test_infeasible_actions_rejected(scene):
    env = make(scene)
    obs = env.reset(EpisodeSpec(scene, 3))
    a = int(np.flatnonzero(obs["mask"])[0])
    env.step(a)
    with pytest.raises(ValueError):
        env.step(a)  # already selected this pass
    infeasible = np.flatnonzero(~env._feasible[0])
    if infeasible.size:
        with pytest.raises(ValueError):
            env.step(int(infeasible[0]))


@pytest.mark.parametrize("w", [1.0, 3.0])
def test_difference_rewards_sum_to_map_error_reduction(scene, w):
    env = make(scene, sensor=PERFECT, acq_fail_prob=0.0, reward={"kind": "err", "fn_weight": w, "unc_coef": 0.0})
    env.reset(EpisodeSpec(scene, 4))
    truth = env._truth_eval[0]
    cost = lambda m: w * (~m & truth).sum() + (m & ~truth).sum()
    c0 = cost(env.est_prob >= 0.5)
    total = 0.0
    for _ in range(env.K):
        _, r, _, _ = env.step(int(np.flatnonzero(env.observation()["mask"])[0]))
        total += r
    assert np.isclose(total, (c0 - cost(env.est_prob >= 0.5)) / (TILE * TILE))


def test_episode_determinism(scene):
    env = make(scene)
    a = run_episode(env, BASELINES["greedy_unc"](), EpisodeSpec(scene, 5))
    b = run_episode(env, BASELINES["greedy_unc"](), EpisodeSpec(scene, 5))
    assert [p["iou"] for p in a["passes"]] == [p["iou"] for p in b["passes"]]


def test_rl_policy_respects_mask_and_checkpoint_roundtrip(scene, tmp_path):
    env = make(scene)
    for rec in (True, False):
        m = ActorCritic(recurrent=rec)
        torch.save(m.state_dict(), tmp_path / "m.pt")
        m2 = ActorCritic(recurrent=rec); m2.load_state_dict(torch.load(tmp_path / "m.pt", weights_only=True))
        ep = run_episode(env, RLPolicy(m2, deterministic=False), EpisodeSpec(scene, 6))  # raises if infeasible
        assert len(ep["passes"]) == env.P


def test_recurrent_update_path_matches_rollout_path():
    """PPO update (fused sequence) must reproduce the logits seen step-by-step during the rollout."""
    torch.manual_seed(0)
    m = ActorCritic(recurrent=True)
    g, v = torch.rand(7, 3, len(GRID_CHANNELS), 16, 16), torch.rand(7, 3, len(VEC_FEATURES))
    mk = torch.rand(7, 3, N_TILES) > 0.3
    fused = m(g, v, mk, m.initial_state(3, "cpu"))[0]
    st, outs = m.initial_state(3, "cpu"), []
    for t in range(7):
        o, _, st = m(g[t:t + 1], v[t:t + 1], mk[t:t + 1], st)
        outs.append(o)
    assert torch.allclose(fused, torch.cat(outs), atol=1e-5)


def test_iou_difference_reward_matches_single_tile_iou_change(scene):
    env = make(scene, sensor=PERFECT, acq_fail_prob=0.0, reward={"kind": "iou_diff", "unc_coef": 0.0})
    for ep_seed in (7, 8):   # second reset must not reuse cached confusion counts
        env.reset(EpisodeSpec(scene, ep_seed))
        for _ in range(env.K - 1):
            env.step(int(np.flatnonzero(env.observation()["mask"])[0]))
        truth = env._truth_eval[0]
        iou = lambda pred: ((pred & truth).sum() + IOU_EPS) / ((pred | truth).sum() + IOU_EPS)
        before = iou(env.est_prob >= 0.5)
        a = int(np.flatnonzero(env.observation()["mask"])[0])
        prob, valid = env._acquire(a, 0)
        prob = env._fuse(prob, a)
        est = env.est_prob.copy()
        r_, c_ = divmod(a, 16)
        sl = (slice(r_ * TILE, (r_ + 1) * TILE), slice(c_ * TILE, (c_ + 1) * TILE))
        est[sl] = np.where(valid, prob, est[sl])
        _, r, _, _ = env.step(a)
        assert np.isclose(r, 100 * (iou(est >= 0.5) - before), atol=1e-9)


def test_map_product_unknown_and_layers(scene):
    env = make(scene)
    env.reset(EpisodeSpec(scene, 9))
    mp = env.map_product()
    assert (mp["flood_class"] == env.UNKNOWN).all() and (mp["age_h"] == 65535).all()
    for _ in range(env.K):
        env.step(int(np.flatnonzero(env.observation()["mask"])[0]))
    mp = env.map_product()
    seen = env.est_valid
    assert seen.any() and set(np.unique(mp["flood_class"][seen])) <= {0, 1}
    assert (mp["flood_class"][~seen] == env.UNKNOWN).all()
    assert mp["confidence"].dtype == np.uint8 and mp["confidence"][~seen].max() == 0


def test_iou_reward_variant_runs_full_episode(scene):
    """Regression: the 'iou' ablation reward crashed at the end of the first pass (KeyError)."""
    env = make(scene, reward={"kind": "iou", "fn_weight": 3.0, "unc_coef": 0.0})
    ep = run_episode(env, BASELINES["random"](), EpisodeSpec(scene, 11))
    assert len(ep["passes"]) == env.P and np.isfinite(ep["return"])
