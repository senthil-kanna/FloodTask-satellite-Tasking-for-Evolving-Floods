# Troubleshooting and issues found during development

| Symptom | Cause | Fix (in code) |
|---|---|---|
| `load_scene` / calibration hangs forever | Reach-zone graph had cycles: a river leaving block A, entering B, re-entering A. The inlet walk looped | `hand.zone_links`: each zone links downstream only through its max-accumulation outlet. Accumulation increases strictly downstream, so the graph is acyclic. Regression test `test_zone_graph_acyclic_on_meandering_river`. Old scenes: `python -m floodtask.data.scenes --relink` |
| Simulated floods ~9x larger than labelled water fraction, insensitive to runoff | (a) 200-cell channel threshold made every small stream HAND=0; (b) any positive storage flooded all HAND~0 cells | Channel threshold chosen on training events (`select_channel_threshold.py`, picked 1000 cells). Overbank storage threshold `loss_mm` in the simulator, calibrated jointly with runoff/inflow scales |
| Imaging a tile lowers map accuracy on average | U-Net false-positive rate (~3% of dry 60 m cells) on mostly dry scenes; "dry" prior was better than a noisy look | Shared estimator fuses observations with a terrain prior P(water/HAND) (prior-shift corrected). Selected on validation |
| PPO does not learn (entropy constant, KL ~1e-9) | Encoder activations shrank ~2.5x per conv layer; tile logits were nearly identical, gradients tiny | GroupNorm after each conv, raw-input skip connection into the action head, head gain 0.1. Pilot KL rose from ~1e-9 to ~5e-3 per update |
| Real-event download `KeyError: 'vh'` | Some 2016 S1 acquisitions are single-polarisation | Only dual-pol (VV+VH) items are used; skipped items are recorded in `acquisitions.json` |
| `WarpOperationError ... IReadBlock failed` during S1 download | Planetary Computer SAS tokens expire after ~1 h; transient HTTP read errors | Re-sign each URL right before reading; retry with backoff; resumable per scene |
| Background job dies after 10 min | Tool/terminal timeout | Use the detached launchers `scripts/train_all.ps1` and `scripts/download_real_events.ps1` (Start-Process) |
| `seg.train` slower than expected | First epoch includes chip caching; CPU contention | fp16 autocast + cudnn.benchmark; ~4 s/epoch afterwards |

## Checks to run after any change
```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m floodtask.data.sen1floods11 --verify
.\.venv\Scripts\python scripts\make_manifest.py --verify
```
