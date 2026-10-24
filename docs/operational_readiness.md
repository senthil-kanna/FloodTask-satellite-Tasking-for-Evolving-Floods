# Operational-readiness assessment (status: RESEARCH PROTOTYPE, NOT OPERATIONAL)

This is not certified, not validated by an independent body, and not authorised for emergency use. The items
below are what an agency (e.g. ASDMA, NDMA, NRSC/ISRO) would need before operational trials.

| Area | Current state in this repository | Gap before operational use |
|---|---|---|
| Input data quality and timeliness | Every observation carries an acquisition time, validity fraction and failure flag. Forecast/gauge-missing flags are explicit | Real-time ingestion (S1/commercial SAR, IMD forecasts, CWC gauges) not implemented; latencies assumed |
| Map uncertainty communication | Per-tile sensor uncertainty (entropy) and staleness are tracked separately in the estimator state | No user-facing uncertainty product; no calibration study of map-level uncertainty on real events |
| Timestamps and freshness | Map age per tile available (`obs_age`) | Product format (GeoTIFF + metadata, OGC services) not implemented |
| Blind spots and failed acquisitions | Failures are logged per decision; never-observed tiles are flagged | Explicit "unknown" class in the delivered map (unobserved tiles are currently shown as not flooded) |
| Audit trail | Every tile decision (pass, slot, tile, failed, age before) is stored per episode (`history` in core-suite `episodes.jsonl`). Configs and checksums are recorded | Tamper-evident logging, signed artefacts, retention policy |
| Data integrity and security | SHA-256 manifests for raw and derived data | Access control, secure update channel for models |
| Model drift and geographic transfer | Event-held-out evaluation (India held out); segmentation IoU on India ~0.53 | Regional fine-tuning with Indian labels; drift monitoring per season and sensor |
| Human review and escalation | Not implemented | Analyst approval of each tasking plan; escalation when uncertainty is high or inputs are missing |
| Fallback tasking | Validation-tuned heuristic baselines (`greedy_unc`, `stalest`) run with the same interface and need no forecasts | Automatic switch-over policy and health checks |
| Integration with a real mission | Abstract budget/swath model | Orbit propagation, agility and slew limits, downlink scheduling, tasking API of an actual operator |
| Independent validation | None | Blind evaluation by an independent team on events and sensors not used here |

## Conditions under which outputs must not be relied on
See docs/limitations.md, section "Conditions under which the system must not be relied on".
