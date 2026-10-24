# System architecture and data flow

```
                 REAL DATA                                     DERIVED / CALIBRATED (train/val only)
 ┌──────────────────────────────┐
 │ Sen1Floods11 S1 + hand labels│──► seg/train.py ──► U-Net (3 seeds) ──► scripts/fit_sensor_model.py ──► sensor_model.json
 │ (11 events, 1 date each)     │        (train events; ckpt+temperature on val)       (val events, 60 m stats)
 └──────────────┬───────────────┘
                │ chip footprints + labels
 ┌──────────────▼───────────────┐
 │ Copernicus DEM GLO-30        │──► data/scenes.py ──► 370 scenes (60 m, 192x192 cells, 16x16 tiles)
 └──────────────────────────────┘     HAND, D8, accumulation, reach zones (acyclic outlet links)
                                       │  select_channel_threshold.py (train) ──► channel_threshold.json
                                       │  fit_hand_prior.py (train)          ──► hand_prior.json (estimator prior)
 ┌──────────────────────────────┐      │
 │ ERA5 daily precip (Open-Meteo)│──► calibrate_sim.py (train) ──► sim_calibration.json (storm, runoff, inflow, overbank)
 └──────────────────────────────┘      │
                                       ▼
                     ┌────────────────────────────────────────────────────────┐
                     │ FloodTaskingEnv (env/tasking_env.py)                    │
                     │  hidden: rain field, reach storages, flood masks (future)│
                     │  exogenous: swaths, failures, sensor noise, forecast err │
                     │  agent obs: 18x16x16 grid + 10 vector features + mask    │
                     │  shared estimator: Bayesian fusion with HAND prior       │
                     │  evaluator: per-pass IoU/F1/... vs hidden truth          │
                     └───────────┬───────────────────────────┬────────────────┘
                                 │ obs / action / reward      │ EpisodeSpec (paired)
              ┌──────────────────▼─────────┐     ┌────────────▼──────────────────────────┐
              │ rl/train_ppo.py            │     │ policies/baselines.py (val-tuned)      │
              │ CNN+GroupNorm → LSTM|MLP → │     │ random, stalest, flooded_first,        │
              │ FiLM → per-tile logits     │     │ forecast, greedy_unc, oracle (non-op)  │
              │ masked categorical, PPO    │     └────────────┬──────────────────────────┘
              └──────────────┬─────────────┘                  │
                             └──────────► eval/runner.py ◄────┘
                                          scripts/run_experiments.py (test scenes, suites)
                                          scripts/run_real_event.py  (ReplayEnv, Assam S1 RTC)
                                                   │ episodes.jsonl (+ audit trail)
                                                   ▼
                                          scripts/analyze.py → tables, stats (eval/stats.py), figures

 ┌──────────────────────────────┐
 │ Sentinel-1 RTC (Planetary    │──► data/real_event.py: download → U-Net seed0 obs; seeds1+2 ∩ VH-Otsu reference
 │ Computer), Assam 2016 / 2022 │                         → env/replay_env.py
 └──────────────────────────────┘
```

## Modules
| Path | Responsibility |
|---|---|
| `floodtask/data/` | Downloads, manifests, scene construction, ERA5, real-event time series |
| `floodtask/hydro/hand.py` | Depression filling, D8, accumulation, HAND, reach zones |
| `floodtask/hydro/simulator.py` | Storm generator and reach-storage flood model (vectorised linear routing) |
| `floodtask/seg/` | U-Net, dataset with nodata handling, training, test evaluation |
| `floodtask/env/` | Sensor model, tasking POMDP, replay env, factory (variants, fixed EpisodeSpecs) |
| `floodtask/policies/baselines.py` | Common `Policy` interface and baselines |
| `floodtask/rl/` | Actor-critic model and PPO trainer |
| `floodtask/eval/` | Metrics, episode runner, paired cluster statistics |
| `scripts/` | Calibration, tuning, training orchestration, experiments, analysis, manifest |

## Information boundaries
- Policies receive only `observation()`. Ground truth is used for (a) the training reward inside the simulator,
  (b) evaluator metrics, and (c) the oracle, which is labelled non-operational. `tests/test_env.py::test_no_future_truth_leakage`
  corrupts all hidden future state and asserts that the observation is unchanged.
- Synthetic forecasts are built from future simulated rain plus error. This is a modelling assumption about NWP skill.
  It is not leakage of the flood state, and it is stress-tested (uninformative / missing forecasts).

## Compute profile (RTX 4050 laptop, measured)
- U-Net training: 5.3 min per seed (80 epochs, fp16 autocast).
- Env reset ~17 ms. A baseline episode (192 decisions) takes about 0.1 s.
- PPO: ~7.5 s per iteration (16 envs x 192 steps); 1.5 M steps take about 1 h per run.
