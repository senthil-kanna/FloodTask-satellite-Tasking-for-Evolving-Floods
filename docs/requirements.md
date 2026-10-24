# Technical requirements and acceptance criteria

Scope: research prototype for decision support in flood-mapping satellite tasking. Target region for evaluation:
**India - Assam, Brahmaputra valley**. Not certified, not operational (see docs/operational_readiness.md).

"Target" values are project targets. Achieved values are reported only in docs/acceptance_report.md and come from
saved result files. Thresholds marked [V] are set from **validation** baseline performance before test evaluation
(record in `results/requirement_thresholds.json`). They are never set or revised from test outcomes.

| ID | Requirement | Metric / evidence | Acceptance threshold (target) | Rationale |
|---|---|---|---|---|
| R1 | Per-pass mapping accuracy | IoU at each of 12 passes, test scenes | Report all passes; no acceptance threshold (descriptive) | Timeliness curve, not a single number |
| R2 | Final flood extent accuracy | Final-pass IoU; area bias | Final IoU of recurrent PPO >= best operational baseline (CI lower bound > -0.01); abs area bias <= 0.25 | Emergency maps are used for area and affected-population estimates |
| R3 | Accuracy per acquired tile | Mean per-pass IoU vs budget (8/16/32 tiles) | PPO curve not below best baseline at any budget (CI overlap allowed) | Efficiency of a scarce resource |
| R4 | Timeliness | Passes to reach IoU >= 0.5; fraction reaching it | PPO median <= best baseline median | Early usable map matters most |
| R5 | Budget utilisation and feasibility | Infeasible selections; tiles used / budget | 0 infeasible selections (hard); 100% budget used | Tasking constraints are hard limits |
| R6 | Robustness to forecast error and missing data | Relative drop in mean per-pass IoU vs operational forecasts | Drop <= 10% under every corruption; with missing forecasts, PPO >= stalest baseline [V] | Forecasts and gauges fail in real disasters |
| R7 | Generalisation | Primary endpoint on unseen events (India primary) | PPO >= best baseline on India scenes (CI reported); direction consistent on >= 2 of 3 test events | Event and geographic transfer |
| R8 | Compute | Decision latency (CPU), model size, peak memory | < 50 ms per tile decision on a laptop CPU; < 5 M parameters; training < 2 h per seed on RTX 4050 | Ground-segment planning cycles are minutes |
| R9 | Reproducibility and audit | Seeds, configs, checksums, per-decision audit log, regenerated tables | All reported numbers regenerate from raw results with one command; data manifest checksums verify | Independent review |
| R10 | Statement of non-reliance conditions | docs/operational_readiness.md | Present and specific | Safety |
| R11 | Segmentation quality | Test-event IoU of U-Net (India reported separately) | >= Otsu-threshold reference on pooled test events | The sensor model is only as good as the segmenter |
| R12 | Reward validity | Correlation between episode return and mean per-pass IoU (validation) | Spearman rho >= 0.7 | Reward must not be exploitable |
| R13 | Hypothesis H1 (confirmatory) | Recurrent PPO vs each comparator, Holm-corrected | Reported as supported only if Holm p < 0.05 AND CI excludes 0 | Scientific claim |

Notes:
- R2 and R6 tolerances (0.01 IoU, 10%) are judged operationally negligible, since they are smaller than the
  reference-mask uncertainty of SAR flood maps. They are fixed here and not tuned.
- R8 latency target: a tasking plan for 16 tiles must be computed well inside a ground-station planning window.
