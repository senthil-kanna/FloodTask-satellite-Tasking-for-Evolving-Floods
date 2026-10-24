# Known limitations

## Data
1. **No multi-date flood labels.** Sen1Floods11 has one date per event (verified). Flood dynamics for training and
   controlled experiments are simulated. Real-event references are model-derived (U-Net ensemble ∩ VH-Otsu) and are
   checked against hand labels on one date only (Assam 2016-08-12).
2. **Forecasts are not archived NWP.** Simulation: synthetic error model. Replay: ERA5 pseudo-forecast (retrospective).
   Any "forecast value" measured here is the value of forecasts with the assumed error structure. It is not the value
   of a specific operational NWP system.
3. **No real river gauges** (registration-gated sources not used). The gauge input is simulated, or missing in replay.
4. **DEM is a surface model** (vegetation and buildings raise HAND). It is resampled to 60 m, so small channels and
   embankments are lost.
5. **Sen1Floods11 licence** is not stated in its repository. The data are used for research and not redistributed.
6. **Few test events** (India, Bolivia, Mekong). Event-level generalisation claims are qualitative. India has 59
   simulated test scenes and 33 non-overlapping replay scenes, but they all come from one region.

## Flood model (simulation)
7. Conceptual HAND reach-storage model, **not hydrodynamic**: no momentum, backwater, levee breaches, embankment failure,
   dam operations, tides, or pluvial ponding away from the drainage network. Water level is uniform within a reach zone.
8. Calibrated to the **distribution of flooded fraction** of training chips (KS 0.13). It is not calibrated to observed
   flood timing, because no multi-date labels exist for training events. Rise and recession rates are therefore
   assumptions (residence 12-72 h, external lag 12-60 h).
9. Storms are Gaussian rain cells. Monsoon-scale persistent rain over weeks (typical for Assam) is only partly represented.

## Sensor and estimator
10. Segmentation errors are emulated with a fitted parametric model: spatially correlated logit noise, uniform over
    terrain. Real U-Net errors are terrain- and land-cover-dependent (e.g. flooded vegetation and urban areas missed,
    radar shadow). On test events the model is **pessimistic** relative to the real U-Net (simulated 60 m IoU 0.57 vs
    real 0.70), so absolute map IoUs in simulation are likely underestimates.
11. The U-Net only ties the classical Otsu threshold on test IoU (0.652 ± 0.037 vs 0.650). India is the hardest event
    (IoU ~0.53).
12. RTC gamma0 (replay) vs GRD sigma0 (training): a fixed a-priori conversion is applied; residual shift is measured
    on 2016-08-12 only.
13. The shared estimator keeps the latest observation per cell. It does not propagate flood dynamics into stale tiles.
    A better estimator would improve every policy and could change the ranking.

## Tasking abstraction
14. Per-tile, budget-limited tasking emulates agile commercial SAR (spotlight/stripmap frames), not Sentinel-1, which
    images whole swaths without tasking. Swath feasibility is a random access band, not real orbit propagation.
    Revisit, latency and failure rates are assumptions (varied in experiments).
15. One decision = one tile, greedy within a pass. Joint optimisation of a pass's tile set is not attempted.

## Statistics
16. The analysis unit is the scene. Scenes within an event share climate and terrain, and the bootstrap stratifies by
    event, but with three test events, event-level variance is poorly estimated.

## Conditions under which the system must not be relied on
- Outside lowland riverine floodplains (flash floods in steep terrain, coastal/storm-surge, urban pluvial flooding).
- When the segmentation model has not been validated on the region and sensor in use.
- When forecasts or gauges are known to be unavailable or degraded, unless the "missing" results were checked.
- For any life-safety decision without a human analyst reviewing the map and its uncertainty layer.
