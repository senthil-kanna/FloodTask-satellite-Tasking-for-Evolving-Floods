"""Conceptual flood-evolution model (SYNTHETIC forcing on REAL terrain).

This is NOT a hydrodynamic model. It is a HAND-based reach-storage model:
  - Rainfall: moving Gaussian storm cells on an extended tile grid (synthetic, magnitudes scaled to ERA5 event totals).
  - Each reach zone z is a linear reservoir: dS_z/dt = c * R_z * A_z + sum_{u -> z} Q_u + Q_ext,z - S_z / k_z.
  - External inflow (upstream basin outside the scene) = unit-hydrograph response to rain over the upstream margin.
  - Water level h_z = V_z^-1(max(S_z - L * A_z, 0)), V_z(h) = sum_cells max(h - HAND, 0) * A_cell, where L (mm) is the
    in-channel/soil storage that must fill before overbank flooding (bankfull / initial-abstraction concept).
  - Flooded cells: HAND < h_zone(cell). Supports expansion and recession (outflow drains storage).
Known omissions (docs/limitations.md): momentum, backwater, levees, pluvial ponding away from channels,
infiltration dynamics, snowmelt, tides, dam operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from floodtask.data.scenes import CELL, OUT as SCENE_DIR, SCENE, TILE

NT = SCENE // TILE           # tiles per side (16)
EXT = 8                      # margin tiles around the scene on the rain grid
NE = NT + 2 * EXT            # extended rain grid side (32)
H_GRID = np.linspace(0, 25, 501)


@dataclass
class SimParams:
    """Episode-level physical parameters; sampled per episode from ranges in configs/env.yaml."""
    runoff_coef: float = 0.5
    residence_h: np.ndarray | None = None      # per zone
    storm_total_mm: float = 120.0              # target scene-mean storm total over the episode window
    n_storms: int = 2
    ext_scale: float = 1.0                     # external basin size relative to in-scene contributing area
    ext_lag_h: float = 36.0
    loss_mm: float = 0.0                       # in-channel + soil storage filled before overbank flooding (per zone area)
    extra: dict = field(default_factory=dict)


@lru_cache(maxsize=512)
def load_scene(name: str) -> dict:
    d = dict(np.load(SCENE_DIR / f"{name}.npz"))
    hand, zone = d["hand"], d["zone"]
    Z = int(zone.max()) + 1
    A = CELL * CELL
    # storage-volume curve per zone
    vol = np.zeros((Z, H_GRID.size))
    area = np.zeros(Z)
    for z in range(Z):
        hz = hand[zone == z]
        area[z] = hz.size * A
        vol[z] = np.maximum(H_GRID[:, None] - hz[None, :], 0).sum(1) * A
    # tile-level zone weights (fraction of each scene tile belonging to each zone)
    tz = np.zeros((Z, NT * NT))
    zt = zone.reshape(NT, TILE, NT, TILE).transpose(0, 2, 1, 3).reshape(NT * NT, -1)
    for t in range(NT * NT):
        zz = zt[t][zt[t] >= 0]
        if zz.size:
            tz[:, t] = np.bincount(zz, minlength=Z) / (TILE * TILE)
    order = _topo_order(d["zone_down"])
    # main river: zone containing the cell with the largest accumulation
    main_zone = int(zone.ravel()[np.argmax(np.where(zone.ravel() >= 0, d["acc"].ravel(), -1))])
    # inlet: walk the main stem upstream, always taking the upstream zone with the largest accumulation
    zone_acc = np.array([d["acc"][zone == z].max() for z in range(Z)])
    inlet, seen = main_zone, {main_zone}
    while True:
        ups = [u for u in range(Z) if d["zone_down"][u] == inlet and u not in seen]
        if not ups:
            break
        inlet = max(ups, key=lambda u: zone_acc[u])
        seen.add(inlet)
    rr, cc = np.nonzero(zone == inlet)
    r, c = rr.mean() / SCENE - 0.5, cc.mean() / SCENE - 0.5
    side = (0 if r < 0 else 2) if abs(r) >= abs(c) else (3 if c < 0 else 1)
    d.update(Z=Z, vol=vol, zone_area=area, tile_zone=tz, order=order, main_zone=main_zone,
             inlet_zone=inlet, upstream_side=side, zone_acc=zone_acc)
    return d


def _topo_order(zone_down: np.ndarray) -> np.ndarray:
    Z = len(zone_down)
    indeg = np.zeros(Z, int)
    for z in range(Z):
        if zone_down[z] >= 0:
            indeg[zone_down[z]] += 1
    order, stack = [], [z for z in range(Z) if indeg[z] == 0]
    while stack:
        z = stack.pop(); order.append(z)
        d = zone_down[z]
        if d >= 0:
            indeg[d] -= 1
            if indeg[d] == 0:
                stack.append(d)
    if len(order) < Z:  # cycles cannot occur on a filled DEM; guard anyway
        order += [z for z in range(Z) if z not in order]
    return np.array(order)


def storm_field(rng: np.random.Generator, hours: int, p: SimParams) -> np.ndarray:
    """Hourly rain [hours, NE, NE] in mm/h on the extended tile grid."""
    tile_km = TILE * CELL / 1000
    yy, xx = np.mgrid[0:NE, 0:NE] * tile_km
    rain = np.zeros((hours, NE, NE), np.float32)
    t = np.arange(hours)[:, None, None]
    for _ in range(p.n_storms):
        t0 = rng.uniform(0, hours * 0.6)
        dur = rng.uniform(6, 48)
        sig = rng.uniform(4, 15)                         # km
        cx, cy = rng.uniform(-5, NE * tile_km + 5, 2)
        speed, ang = rng.uniform(0, 3), rng.uniform(0, 2 * np.pi)  # km/h
        vx, vy = speed * np.cos(ang), speed * np.sin(ang)
        tt = t - t0
        env = np.exp(-0.5 * ((tt - dur / 2) / (dur / 4)) ** 2) * (tt > 0) * (tt < dur)
        dx = xx[None] - (cx + vx * tt); dy = yy[None] - (cy + vy * tt)
        rain += (env * np.exp(-0.5 * (dx**2 + dy**2) / sig**2)).astype(np.float32)
    scene_mean_total = rain[:, EXT:EXT + NT, EXT:EXT + NT].sum(0).mean()
    rain *= p.storm_total_mm / max(scene_mean_total, 1e-6)
    return rain


def upstream_mask(side: int) -> np.ndarray:
    m = np.zeros((NE, NE), bool)
    if side == 0: m[:EXT, :] = True
    elif side == 1: m[:, -EXT:] = True
    elif side == 2: m[-EXT:, :] = True
    else: m[:, :EXT] = True
    return m


def simulate(scene: dict, rain: np.ndarray, p: SimParams) -> np.ndarray:
    """Return water level per zone per hour [hours, Z]."""
    hours, Z = rain.shape[0], scene["Z"]
    rain_tiles = rain[:, EXT:EXT + NT, EXT:EXT + NT].reshape(hours, -1)          # mm/h
    rain_zone = rain_tiles @ scene["tile_zone"].T / np.maximum(scene["tile_zone"].sum(1), 1e-9)  # mm/h per zone
    A = scene["zone_area"]
    # external inflow: gamma unit hydrograph on upstream-margin rain
    up = rain[:, upstream_mask(scene["upstream_side"])].mean(1)                    # mm/h
    k = np.arange(hours)
    uh = (k / p.ext_lag_h) ** 2 * np.exp(-2 * k / p.ext_lag_h); uh /= uh.sum()
    ext_area = p.ext_scale * A.sum()
    q_ext = np.convolve(up, uh)[:hours] / 1000 * p.runoff_coef * ext_area      # m3/h
    zd, main = scene["zone_down"], scene["inlet_zone"]
    res = p.residence_h
    # Within each hour, water x_z available in zone z = S_z + inflow_z + sum_{u->z} x_u / r_u (upstream outflow
    # routed in the same hour). Linear in (S + inflow): x = (I - U)^-1 (S + inflow), U[z,u] = 1/r_u if u -> z.
    U = np.zeros((Z, Z))
    for u in range(Z):
        if zd[u] >= 0:
            U[zd[u], u] = 1 / res[u]
    M = np.linalg.inv(np.eye(Z) - U)
    inflow = rain_zone / 1000 * p.runoff_coef * A[None]
    inflow[:, main] += q_ext
    keep = 1 - 1 / res
    S = np.zeros(Z); storage = np.zeros((hours, Z))
    for h in range(hours):
        S = (M @ (S + inflow[h])) * keep
        storage[h] = S
    level = np.empty((hours, Z), np.float32)
    for z in range(Z):
        excess = np.maximum(storage[:, z] - p.loss_mm / 1000 * A[z], 0.0)
        level[:, z] = np.interp(excess, scene["vol"][z], H_GRID)
    return level


def flood_mask(scene: dict, level_row: np.ndarray) -> np.ndarray:
    zone = scene["zone"]
    lv = np.where(zone >= 0, level_row[np.maximum(zone, 0)], 0.0)
    return (scene["hand"] < lv) & (lv > 0)
