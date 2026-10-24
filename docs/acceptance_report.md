# Final acceptance report

Status: **research prototype, not operational.** Generated 2026-10-24 from saved result files after Stage C
(`scripts/stage_c_evaluate.ps1`). Every number below is read from a file named in the Evidence column; nothing is
estimated. Thresholds are the ones fixed in `docs/requirements.md` before test evaluation.

## Verdict

| | |
|---|---|
| Requirements met | R1, R3, R4, R5, R7, R8, R9, R10, R11, R12 |
| Partly met | R2 (final IoU met, area bias not met on pooled test events), R13 (H1 supported against all 5 conventional baselines, not against the two PPO ablations) |
| Not met | R6 (missing hydrological observations cost 11.0% of mean per-pass IoU, limit 10%) |

**Answer to the research question.** In the simulated test events, recurrent PPO produces more accurate flood maps
than every conventional tasking strategy under the same acquisition limits: +0.043 mean per-pass IoU over the best
baseline (greedy uncertainty), 95% CI +0.035 to +0.051, Holm p = 2.1e-14, positive for all 5 training seeds and all 3
test events. The two parts that give the method its name are **not** shown to add value: feed-forward PPO is slightly
better than recurrent PPO (-0.002, CI -0.004 to -0.000), and removing forecast inputs changes nothing measurable
(-0.000, CI -0.003 to +0.002). On the real Assam replays, PPO does not beat the simple heuristics (see "Real-event
replay" below). The claim supported by the evidence is therefore "learned tasking beats hand-built tasking in
simulation", not "forecast-aware memory improves tasking".

## Requirement by requirement

All test results are on the 95 held-out test scenes (Bolivia, India, Mekong events; 378 paired episodes per policy,
5 training seeds averaged) unless stated. PPO = forecast-aware recurrent PPO, best-validation checkpoint per seed.

| ID | Requirement and threshold | Achieved | Status | Evidence |
|---|---|---|---|---|
| R1 | Per-pass IoU at all 12 passes, descriptive | Reported for every policy and pass, all test events and India only | **Met** | `results/analysis/summary.md`, `results/figures/per_pass_iou_*.png` |
| R2 | Final IoU of PPO >= best operational baseline (CI lower bound > -0.01); abs area bias <= 0.25 | Final IoU 0.354 vs 0.303 (greedy uncertainty): +0.050, CI +0.041 to +0.060. Mean final area bias +2.56 on all test events (fails); +0.12 on India (passes). The pooled mean is dominated by small-flood episodes, where a few extra cells give large relative bias: the median is +0.019 | **Partly met** | `stats_core_all_test_events.json` (secondary), `core_table_*.csv` |
| R3 | PPO not below best baseline at 8 / 16 / 32 tiles per pass | vs greedy uncertainty: +0.044 (CI +0.036 to +0.053) at 8, +0.043 (+0.035 to +0.051) at 16, +0.023 (+0.018 to +0.029) at 32 | **Met** | `results/analysis/stats_conditions.json`, `budget_curve.png` |
| R4 | PPO median passes to IoU 0.5 <= best baseline median | Median 13 (censored) for both: most episodes never reach 0.5 within 12 passes. PPO reaches it in 17.7% of episodes vs 4.2%; paired mean -0.69 passes (CI -0.97 to -0.45) | **Met** (as written; weak test because medians are censored) | `stats_core_all_test_events.json` (secondary), `core_table_all_test_events.csv` |
| R5 | 0 infeasible selections; 100% budget used | 6,080 core-suite episodes: 0 infeasible selections (the runner raises on any infeasible tile and none was raised); every pass made exactly 16 selections. Failed acquisitions (simulated sensor failures) are logged per decision and are not counted as infeasible | **Met** | `results/experiments/core/episodes.jsonl` (`history`), `floodtask/eval/runner.py` |
| R6 | Drop <= 10% under every forecast corruption; with missing forecasts PPO >= stalest | Change vs operational forecasts (0.268): noisy -0.1%, biased high -0.7%, biased low +0.8%, displaced +0.1%, uninformative -5.7%, missing forecasts -2.2%, **missing rain and gauge observations -11.0%**. Missing forecasts: PPO 0.263 vs stalest 0.133 | **Not met** (one condition over the limit) | `results/analysis/conditions_mean_iou.csv`, `forecast_robustness.png` |
| R7 | PPO >= best baseline on India (CI reported); same direction on >= 2 of 3 events | India: +0.040 vs greedy uncertainty, CI +0.031 to +0.049, Holm p = 1.1e-9. Direction positive on 3 of 3 events (Bolivia +0.058, India +0.040, Mekong +0.042) | **Met** | `stats_core_India.json`, `stats_core_all_test_events.json` |
| R8 | < 50 ms per tile decision on a laptop CPU; < 5 M parameters; training < 2 h per seed | Isolated benchmark, 1 CPU thread: 5.9 ms mean, 9.1 ms p95 per decision (94 ms for a 16-tile plan). 769,474 parameters. Longest training run 103 min. Peak memory 477 MB. In-suite latencies (41 ms mean) are inflated by parallel workers and are not used here | **Met** | `results/latency_bench.json` (`scripts/bench_latency.py`), `results/rl/*/seed*/train_log.json` |
| R9 | All numbers regenerate with one command; manifest checksums verify | `python scripts/analyze.py` regenerated all tables and figures on 2026-10-24. `python scripts/make_manifest.py --verify`: 678 files, 0 mismatches | **Met** | `results/data_manifest.json`, `results/logs/stage_c.log` |
| R10 | Non-reliance conditions present and specific | Present | **Met** | `docs/operational_readiness.md`, `docs/limitations.md` |
| R11 | U-Net test IoU >= Otsu reference on pooled test events | Pooled test IoU 0.652 (mean of 3 seeds, SD 0.036) vs Otsu 0.650. India: 0.530 vs 0.529. Seed 1 alone (0.610) is below Otsu | **Met** (by a small margin) | `results/segmentation/eval_test.json` |
| R12 | Spearman(return, mean per-pass IoU) >= 0.7 on validation | 0.83 episode level, 1.00 policy level (reward with missed-flood weight 3). Test-set check: 0.57 | **Met** on validation, as specified. Lower on test, reported | `results/reward_validity_val.json`, `summary.md` |
| R13 | H1 supported only if Holm p < 0.05 and CI excludes 0, per comparison | Supported against random (+0.153), stalest (+0.135), flooded-first (+0.085), forecast-following (+0.087), greedy uncertainty (+0.043): all Holm p <= 2.1e-14. **Not supported** against feed-forward PPO (-0.002, CI -0.004 to -0.000, Holm p 0.011, wrong direction) or PPO without forecasts (-0.000, CI -0.003 to +0.002, Holm p 0.40). Mixed-effects cross-check agrees in sign and significance for all 7 | **Partly met** | `stats_core_all_test_events.json` |

## Real-event replay (descriptive, no acceptance threshold)

33 Sentinel-1 scenes over the Brahmaputra valley. Reference masks are model-derived (U-Net ensemble AND VH-Otsu
agreement); on the 2016-08-12 hand-labelled date the reference itself scores IoU 0.337 against hand labels and the
deployed U-Net 0.090. The forecast is an ERA5 pseudo-forecast built after the event, not an archived forecast.

| Event, forecast on | PPO mean per-pass IoU | Best heuristic | Gap |
|---|---|---|---|
| Assam 2016 (4-8 passes) | 0.077 | Flooded-first 0.128 | -0.051 |
| Assam 2022 (16-25 passes) | 0.135 | Flooded-first 0.223 | -0.087 |

The simulated-test advantage does **not** carry over to these replays. Likely contributors, not tested separately:
reference-mask noise of the same size as the differences, a real flood regime unlike the simulator's, and a much
longer pass sequence (25 vs 12) than PPO was trained on. Per-scene maps, tasking plans and scores are in the map
viewer (`results/viewer/index.html`, built by `scripts/build_viewer_data.py`) and in `results/products/`.

## Flood outlook (added after acceptance testing, not covered by R1-R13)

`floodtask/forecast/outlook.py` predicts the chance that each 60 m cell is flooded 4, 6, 12 and 24 h after a pass. It
runs 48 copies of the flood model with physical settings drawn from the training prior, driven by rain observed so far
and the rain forecast issued at the pass, and weights each copy by how well it matches the current map. The weight
temperature (0.02) was chosen on validation scenes (`results/outlook_val.json`, lowest Brier score at every horizon)
before the test run. "Persistence" (the current map unchanged) is the reference.

| Test | Outlook IoU | Persistence IoU | Evidence |
|---|---|---|---|
| Simulated test scenes, +4 h | 0.646 | 0.217 | `results/outlook_test.json` (paired gain +0.43, CI +0.39 to +0.47, 91 scenes) |
| Simulated test scenes, +6 h | 0.655 | 0.215 | same (+0.45, CI +0.41 to +0.49, 91 scenes) |
| Simulated test scenes, +12 h | 0.648 | 0.205 | same (+0.44, CI +0.40 to +0.47, 92 scenes) |
| Simulated test scenes, +24 h | 0.675 | 0.194 | same (+0.49, CI +0.45 to +0.52, 93 scenes) |
| Assam 2016, next real image (median lead 19 days) | 0.059 | 0.036 | `results/outlook_real.json` (143 cases; outlook better in 63%) |
| Assam 2022, next real image (median lead 5 days) | 0.122 | 0.108 | same (672 cases; outlook better in 59%) |

The simulated scores are a best case: the simulated truth is produced by the same flood model, so the outlook only
has to recover parameters and rain, not model error. On the test scenes the unweighted ensemble is as good or slightly
better (+4 h IoU 0.671 vs 0.646), so weighting by the current map adds nothing there and terrain does most of the work.
On real data the gain over persistence is small and absolute accuracy is low; the real images are days apart, so the
4-24 h horizons cannot be checked against real observations at all. The outlook is shown in the map viewer; it is not
used by any tasking policy and is not validated for operational use.

## Deviations and gaps found during acceptance

1. `results/requirement_thresholds.json` was never written. The only threshold marked [V] (R6, "PPO >= stalest with
   missing forecasts") is a comparison rather than a number, so no value was needed, but the file the requirements
   document promises does not exist. It is not created now, because it would postdate the test results.
2. The 13 runs planned for machine B were trained on machine A (the same RTX 4050 laptop) after the second machine
   failed. Settings and seeds are unchanged (`configs/split.yaml`).
3. Stage C's analysis step first failed on a syntax error in `scripts/analyze.py` (a line break inside a string). It
   was fixed and re-run on 2026-10-24; no result changed because the analysis reads the saved episodes.
4. The R8 latency figure comes from `scripts/bench_latency.py`, added after test evaluation because the in-suite
   timings ran next to parallel workers. It measures speed only and does not affect any accuracy result.
5. GPU training is not bit-deterministic (cuDNN, fp16). Evaluation of saved checkpoints is deterministic.

## What an independent reviewer should check first

- Whether the simulated flood regime is representative: the real-event gap is the main threat to the headline result.
- The area-bias definition (relative to true flooded area) before using R2 for policy: it is unstable for small floods.
- Re-run `python scripts/make_manifest.py --verify`, then `python scripts/analyze.py`, and compare with this report.
