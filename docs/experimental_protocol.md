# Experimental protocol (predeclared)

Status: written **before** any test-event evaluation of tasking policies. Freeze record (UTC timestamp + SHA-256)
in `results/protocol_freeze.json`. Any later deviation is listed in the "Deviations" section at the bottom with its reason.

## 1. Data partitions
| Partition | Events | Used for |
|---|---|---|
| Train | Ghana, Paraguay, USA, Pakistan, Sri-Lanka, Spain | U-Net fitting, normalisation statistics, simulator calibration, PPO training |
| Validation | Somalia, Nigeria | U-Net checkpoint + temperature, sensor-model fit, baseline tuning, PPO checkpoint selection, PPO variant selection |
| Test | **India (primary; Assam, Brahmaputra valley)**, Bolivia, Mekong | Final evaluation only; evaluated once, after all choices above are frozen |

Target region: **India - Assam, Brahmaputra valley** (chosen by the project owner's request for an India focus;
the Sen1Floods11 India chips lie at 92.4-94.0 E, 25.7-27.3 N). India is never used for training, tuning or selection.
Splits are by **event**; a scene never mixes events; scenes are built from DEM windows around chip locations.

## 2. Primary endpoint
**Mean per-pass IoU over flood-present passes**: IoU of the shared map estimate against the hidden flood state at
each pass's evaluation time (pass time + 3 h processing delay), averaged over passes whose true flooded fraction is
>= 0.1% (>= 37 cells). Episodes without any flood-present pass are excluded from IoU endpoints and counted; false-alarm
area is reported for all passes.
Rationale: it rewards both accuracy and timeliness. Secondary endpoints: final-pass IoU, F1, precision,
recall, pixel error, area bias, boundary F1 (1-cell tolerance), time-to-threshold, revisits, coverage.

IoU conventions: see `floodtask/eval/metrics.py` (IoU = 1 when truth and prediction are both empty).
Time-to-threshold: first pass at which IoU >= 0.5; episodes that never reach it are assigned 13 (censored),
and the fraction reaching the threshold is reported. The threshold value 0.5 is fixed here and not revised.

## 3. Primary hypothesis and comparisons (Holm-corrected family, alpha = 0.05)
H1: Forecast-aware recurrent PPO achieves higher mean per-pass IoU than the best-on-validation conventional
baseline. Family of paired comparisons on the primary endpoint, core suite:
1. recurrent PPO vs random
2. recurrent PPO vs stalest (uniform coverage)
3. recurrent PPO vs flooded_first
4. recurrent PPO vs forecast-following
5. recurrent PPO vs greedy_unc (tuned greedy uncertainty)
6. recurrent PPO vs feed-forward PPO (value of memory)
7. recurrent PPO vs recurrent PPO without forecast inputs (value of forecasts)

All other analyses (ablations, robustness, generalisation, real-event) are **secondary/exploratory**: they get
CIs and uncorrected p-values. They do not support confirmatory claims.

## 4. Units, pairing and statistics
- Every policy runs on identical `EpisodeSpec`s. Each spec fixes the scene, episode seed, physics, storms, swaths,
  failures, sensor noise seeds and forecast errors, so comparisons are paired.
- Test set: every test scene x 4 episode seeds.
- RL: 5 independent training seeds (0-4); per episode the RL metric is averaged over seeds; per-seed effects are
  reported separately with the number of seeds showing a positive effect.
- Analysis unit = scene (episode differences averaged within scene). 95% CI by bootstrap over scenes
  stratified by event (10,000 resamples). Test: Wilcoxon signed-rank on scene-level differences. Effect size:
  Cohen's d_z. Per-event differences are always reported. Pixels are never treated as samples.
- Only 3 test events exist, so event-level generalisation claims stay qualitative.
- India results are reported separately as the primary regional result: same statistics restricted to India scenes,
  bootstrap over India scenes. Pooled 3-event results are reported alongside.

## 5. Model selection rules (frozen)
- U-Net: best validation IoU epoch; temperature fitted on validation pixels.
- PPO: checkpoint with best validation mean per-pass IoU (32 fixed validation episodes), evaluated every 25 iterations.
- PPO "main" variant: `recurrent` (reward = per-tile cost-weighted map-error reduction, missed-flood weight 3). Alternative rewards
  (`recurrent_iou`, `recurrent_unc`) are ablations; if one beats `recurrent` on validation, that is reported, but
  the confirmatory family stays on `recurrent` as predeclared here.
- Baselines: weights grid-searched on 48 validation episodes using the primary endpoint.

## 6. Secondary suites
- Ablation: no forecast inputs, feed-forward, uncertainty bonus (coef 0.1), IoU-delta reward.
- Forecast robustness (test scenes): accurate, noisy, biased low (x0.4), biased high (x2.5), displaced (4-6 tiles),
  uninformative (climatology), missing; plus missing gauge + rain observations. Policies are trained on the
  "operational" forecast-error mix and not retrained.
- Segmentation quality: perfect, calibrated (default), degraded sensor models.
- Budget: 8 / 16 / 32 tiles per pass. Revisit: 6 / 12 / 24 h. Latency: 0 / 1 pass.
- Flood-size strata: post-hoc tertiles of true flooded fraction at mid-episode.
- Real events (India test scenes, Sentinel-1 RTC time series): **Assam 2016** (Jul-Sep, 11 dates; includes the 2016-08-12
  hand-label date, used to check the reference masks) and **Assam 2022** (May-Aug, 25 dates). All policies frozen;
  per-scene curves; descriptive statistics only (one region, two seasons).

## 7. Requirements thresholds
Thresholds for acceptance (docs/requirements.md) were set from application needs and validation baselines
before test evaluation. They are not changed after test results are seen.

## Deviations
- 2026-10-20, before any test evaluation and before any model finished training: target region changed to India
  at the project owner's request. India moved from train to primary test, Spain from test to train; real-event replay
  changed from Bolivia 2018 to Assam 2016 and 2022. The protocol was re-frozen (new hash in results/protocol_freeze.json).
- 2026-10-20, before any test evaluation of tasking policies, based on VALIDATION evidence only:
  (a) Primary endpoint restricted to flood-present passes. 20% of validation passes had zero flood, where IoU is
      degenerate (0 or 1).
  (b) Main reward: missed-flood weight 3 in the per-tile error reward. With weight 1, Spearman(return, endpoint) on
      validation was 0.77 (episode level) and 0.66 (policy level). Weight 3 gave 0.83 and 1.00. Candidates 1/3/10/30 and an
      IoU difference reward (0.49, rejected) are recorded in results/reward_validity_val.json.
  (c) Shared estimator: Bayesian fusion with a terrain prior P(water|HAND) fitted on training events. It raised
      validation IoU for every policy.
  (d) Simulator: HAND channel threshold (training-event AUC) and an overbank storage threshold (training-event
      flood-fraction calibration). Scenes were rebuilt.
  Requirement R12 is evaluated with the new endpoint.
- PPO hyperparameters (gamma, lr, entropy coefficient, epochs) were selected by a 5-config validation sweep of
  feed-forward PPO (400k steps; results/ppo_sweep.json; chosen gamma 0.9, lr 1e-3, ent 0.003, 8 epochs). The same
  settings are used for all variants. Recurrent PPO was not separately tuned, which is conservative for H1.
- Training is split across two machines (configs/split.yaml); all checkpoints are evaluated on machine A.
