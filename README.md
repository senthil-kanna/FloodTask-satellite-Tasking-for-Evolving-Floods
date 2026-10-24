# Forecast-Aware Reinforcement Learning for Multi-Pass Satellite Imaging of Evolving Floods

Research prototype. It selects which 720 m tiles to image at each satellite pass, under a fixed budget, to keep
a flood map accurate and current. **Target evaluation region: Assam, India (Brahmaputra valley).**
It is not an operational or certified system (see `docs/operational_readiness.md`).

## What is real and what is simulated
| Component | Status |
|---|---|
| SAR flood segmentation (U-Net) | Trained/tested on real Sen1Floods11 Sentinel-1 chips with hand labels, split by event |
| Terrain (HAND, drainage, reach zones) | Real Copernicus DEM GLO-30 |
| Flood evolution for training/controlled tests | **Simulated**: conceptual HAND reach-storage model with synthetic storms (scaled to ERA5 totals) |
| Forecasts, gauges | **Synthetic** with documented error models (no archived forecasts were available; see gap G1) |
| Real-event test | Real Sentinel-1 RTC time series over Assam (2016: 11 dates, 2022: 25 dates); reference masks are model-derived |

## Tested configuration
Windows 11, Python 3.12.6, PyTorch 2.5.1+cu124, NVIDIA RTX 4050 Laptop (6 GB), 16 logical CPUs, 24 GB RAM.
Exact package versions: `requirements-lock.txt`.

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\pip install -r requirements-lock.txt
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\python -m pytest            # unit tests (HAND, metrics, env leakage/feasibility/reward, stats)
```

## Full pipeline (run from repo root, inside the venv)
| Stage | Command | Output |
|---|---|---|
| 2 Data | `python -m floodtask.data.sen1floods11` (+ `--verify`) | `data/raw/sen1floods11`, SHA-256 manifest |
| 2 Data | `python -m floodtask.data.rainfall` | `data/processed/era5_event_rainfall.json` |
| 2 Data | `python -m floodtask.data.scenes` | DEM tiles, `data/processed/scenes/*.npz`, `scenes.json` |
| 3 Segmentation | `python -m floodtask.seg.train --seed {0,1,2}` | `results/segmentation/seed*/` |
| 3 Segmentation | `python -m floodtask.seg.evaluate --seeds 0 1 2` (test, once) | `results/segmentation/eval_test.json` |
| 4 Environment | `python scripts/fit_sensor_model.py` | `results/sensor_model.json` |
| 4 Environment | `python scripts/calibrate_sim.py` | `results/sim_calibration.json` (HAND-AUC validation included) |
| 5 Baselines | `python scripts/tune_baselines.py` (validation only) | `results/baseline_params.json` |
| 5 RL | `python scripts/train_all.py` (5 variants x 5 seeds) | `results/rl/<variant>/seed*/` |
| 6 Experiments | `python scripts/run_experiments.py --suite all` | `results/experiments/*/episodes.jsonl` |
| 7 Real event | `python -m floodtask.data.real_event --event assam2016 --step download` then `--step rain`, `--step predict`; `python scripts/run_real_event.py --event assam2016` (same for `assam2022`) | `results/experiments/real_assam*/` |
| 7 Products | `python scripts/export_maps.py --event assam2016` (deployed agent = best-validation recurrent PPO seed) | `results/products/<event>/<policy>/<scene>/<date>_map.tif` (flood class incl. unknown, confidence, data age) + `<date>_plan.geojson` |
| 7 Planning tool | `python scripts/export_maps.py --event assam2016 --scene India_00 --plan-date 2016-08-12` | Prints the 16 tiles the agent tasks on that date (lon/lat, selection probability) |
| 8 Reporting | `python scripts/analyze.py`; `python scripts/make_manifest.py` | `results/analysis/` (incl. mixed-effects cross-check), `results/figures/`, manifest |
| 8 Compute check | `python scripts/bench_latency.py` (isolated, 1 CPU thread) | `results/latency_bench.json` (R8) |
| 8 Flood outlook | `python scripts/eval_outlook.py --split val` (choose weighting), then `--split test` | `results/outlook_val.json`, `results/outlook_test.json` (outlook vs persistence, simulated truth) |
| 8 Map viewer | `python scripts/build_viewer_data.py` (replays PPO + greedy on Assam 2016/2022 and the simulated Bolivia/India/Mekong test scenes, adds outlooks), then serve `results/viewer` (e.g. `python -m http.server -d results/viewer`) | `results/viewer/data/*`, `results/outlook_real.json` (outlook vs persistence at the next real image) |
| 8 Results tabs | `python scripts/export_results_data.py` | `results/viewer/data/results/*.json` (Results, Robustness, Real events, Training tabs of the viewer) |
| Visual check | `python scripts/visualize_episode.py --policy ppo:recurrent:0 --scene India_00` | `results/figures/episodes/*.png` |

**Shortcut, two machines:** Stage B = `scripts\stage_b_train.ps1` (machine A) + `run_machine_b.ps1` from `dist\floodtask_train_bundle.zip`
(machine B, see `docs/run_on_second_machine.md`), merge with `scripts\merge_machine_b.py`; Stage C = `scripts\stage_c_evaluate.ps1`.

## Documentation
`docs/requirements.md`, `docs/architecture.md`, `docs/dataset_inventory.md`, `docs/formulation.md`,
`docs/experimental_protocol.md` (predeclared; frozen hash in `results/protocol_freeze.json`), `docs/limitations.md`,
`docs/operational_readiness.md`, `docs/acceptance_report.md`. (Research paper: deferred until requested.)

## Reproducibility notes
Seeds are fixed and recorded, but cuDNN kernels and fp16 autocast are not bit-deterministic, so GPU-trained weights can
differ slightly between runs. Evaluation given fixed checkpoints is deterministic (tested). Every result JSON carries
the software versions, GPU, timestamp and git commit (if any).
