# Problem formulation (POMDP)

Notation: scene grid $\mathcal{X}$ of $192\times192$ cells (60 m), partitioned into $N=256$ tiles $\mathcal{T}_1..\mathcal{T}_N$
of $12\times12$ cells (720 m). Passes $p=0..P-1$ at times $t_p = t_0 + p\,\Delta$ ($P=12$, $\Delta=12$ h). Budget $K=16$ tiles per pass.
One decision step selects **one tile**; a pass is $K$ consecutive steps, so an episode has $PK=192$ steps.

## Hidden state $s$
- Reach storages $S_z(t)$ and water levels $h_z(t)=V_z^{-1}(S_z(t))$ for reach zones $z$ (hourly).
- Flood state $F_t(x)=\mathbb{1}[\mathrm{HAND}(x) < h_{z(x)}(t)]$.
- Full rainfall field $R(t,\cdot)$ (past and future), episode physics $\theta$ (runoff coefficient, residence times,
  external-basin scale and lag), exogenous pre-sampled acquisition feasibility $A_p\subseteq\{1..N\}$, failures, sensor
  noise seeds, forecast-error parameters.
- Agent-independent: actions do not change the flood. They change only what is known.

## Transition model
$S_z(t+1)=\big[(I-U)^{-1}(S(t)+q(t))\big]_z\,(1-1/r_z)$, with $q_z(t)=c\,R_z(t)A_z + \mathbb{1}[z=z_{in}]Q_{ext}(t)$,
$U_{zu}=1/r_u$ if $u\to z$. $Q_{ext}$ is a gamma unit-hydrograph response to upstream-margin rain. Within a pass, the
step only updates the pass's selection set. After the $K$-th step, acquisitions are processed and the pass advances.

## Observation model
Acquiring tile $i$ at pass $p$ (if $i\in A_p$ and not failed) yields, after `latency_passes`, a probability map
$\hat\pi_i = \sigma\!\big(a(2\,G_b*F_{t_p}-1)+\beta+\sigma_n\,\epsilon_{corr}\big)$ on valid cells. Parameters are fitted to the U-Net
(docs/segmentation.md). The agent's observation $o$ at each step contains only decision-time information:

| Grid channel (16x16) | Meaning |
|---|---|
| est_flood_frac, est_mean_prob | current shared map estimate per tile |
| sensor_uncertainty | mean binary entropy of the last observation of the tile (confidence) |
| obs_age, never_observed | time since last delivered observation (staleness), kept separate from confidence |
| last_valid_frac, last_acq_failed | data-quality flags |
| fcst_24h, fcst_72h, fcst_missing | rain forecast issued at $t_p$ (synthetic error model; see below) |
| rain_obs_24h | observed rain over the past 24 h (noisy) |
| hand_mean, hand_lt1, hand_lt3, log_acc, drainage_frac | static terrain/drainage context |
| feasible, selected_this_pass | constraint state |

Vector: pass fraction, remaining budget, gauge level and 24 h change (main reach, noisy) + missing flag,
upstream-basin forecast 24/72 h, forecast/rain-missing flags, latency.

**Stale vs uncertain.** `sensor_uncertainty` describes how confident the segmentation was when the tile was seen.
`obs_age` describes how old that information is. A tile can be confidently observed and stale at the same time.
The uncertainty used for the bonus and the greedy baseline combines both:
$u_i = 1-(1-H_i)\,e^{-\mathrm{age}_i/\tau}$; never-observed tiles have $u_i=1$.

## Action space and constraints
$a\in\{1..N\}$ subject to $a\in A_p$ and $a\notin$ (tiles already selected in pass $p$). Infeasible actions are masked
(logits $=-10^9$) and rejected by the environment with an exception (tested).

## Map estimator (shared by all policies)
$\hat m(x)$ = latest valid delivered probability for $x$; prior $0$ (not flooded) before any observation; nodata keeps
the previous value. Map $=\mathbb{1}[\hat m\ge 0.5]$. The estimator is fixed, so policies differ only in which tiles they select.

## Reward
Default (difference reward, per selection of tile $i$ at pass $p$, delivered at $d=p+L$):
$r = \frac{1}{|\mathcal{T}_i|}\Big(\sum_{x\in\mathcal{T}_i}\mathbb{1}[\hat M^{-}(x)\ne F_{\tau_d}(x)]-\mathbb{1}[\hat M^{+}(x)\ne F_{\tau_d}(x)]\Big)$
where $\hat M^{\mp}$ is the tile map before/after integrating the observation and $\tau_d=t_d+3$ h. With $L=0$ the
rewards of a pass sum exactly to the global pixel-error reduction (unit-tested). Ground truth enters **only** the
reward, which is computed by the simulator during training. Evaluation and inference never use it.
Ablations: `iou` (global IoU change at the end of the pass) and `+unc` ($+\lambda(u_i^{-}-u_i^{+})$, $\lambda=0.1$).

## Termination
Fixed horizon after $P$ passes (the response window). No early termination, so every policy gets the same exposure.

## Forecast inputs (honesty statement)
In simulation, forecasts are **synthetic**: the future simulated rain multiplied by lognormal errors, with a
per-episode bias, lead-time-dependent noise and spatial displacement. The "operational" mix is used in training.
Stress modes are listed in docs/experimental_protocol.md. In real-event replays, forecasts are ERA5 future accumulation
times errors. That is a **pseudo-forecast** and was not available in real time. Results are also reported without forecasts.
