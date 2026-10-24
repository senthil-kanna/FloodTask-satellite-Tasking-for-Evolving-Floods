"""Terrain hydrology on a regular grid: depression filling, D8 flow, accumulation, HAND, reach zones.

Methods:
- Priority-flood depression filling with epsilon gradient (Barnes, Lehman & Mulla 2014).
- D8 steepest-descent flow directions (O'Callaghan & Mark 1984).
- HAND: elevation above the drainage cell reached by following D8 flow (Renno et al. 2008; Nobre et al. 2011).
Flow leaves the grid at its border, so the computation window should include a buffer around the scene.
"""
from __future__ import annotations

import heapq

import numpy as np

# D8 neighbour offsets
DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
DIST = np.sqrt(DR**2 + DC**2)


def fill_depressions(dem: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    H, W = dem.shape
    filled = dem.astype(np.float64).copy()
    done = np.zeros((H, W), bool)
    heap = []
    for r in range(H):
        for c in (0, W - 1):
            heapq.heappush(heap, (filled[r, c], r, c)); done[r, c] = True
    for c in range(1, W - 1):
        for r in (0, H - 1):
            heapq.heappush(heap, (filled[r, c], r, c)); done[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for k in range(8):
            rr, cc = r + DR[k], c + DC[k]
            if 0 <= rr < H and 0 <= cc < W and not done[rr, cc]:
                done[rr, cc] = True
                if filled[rr, cc] <= z:
                    filled[rr, cc] = z + eps
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def flow_directions(filled: np.ndarray) -> np.ndarray:
    """Flat index of the downstream cell, or -1 if flow leaves the grid (border / no lower neighbour)."""
    H, W = filled.shape
    pad = np.pad(filled, 1, constant_values=-np.inf)  # border cells drain outward
    best = np.zeros((H, W)); arg = np.full((H, W), -1)
    for k in range(8):
        nb = pad[1 + DR[k]:1 + DR[k] + H, 1 + DC[k]:1 + DC[k] + W]
        slope = (filled - nb) / DIST[k]
        upd = slope > best
        best[upd] = slope[upd]; arg[upd] = k
    rr, cc = np.indices((H, W))
    tr, tc = rr + np.where(arg >= 0, DR[arg], 0), cc + np.where(arg >= 0, DC[arg], 0)
    inside = (arg >= 0) & (tr >= 0) & (tr < H) & (tc >= 0) & (tc < W)
    return np.where(inside, tr * W + tc, -1).ravel()


def accumulation(filled: np.ndarray, down: np.ndarray) -> np.ndarray:
    order = np.argsort(-filled.ravel(), kind="stable")  # upstream (high) first
    acc = np.ones(filled.size)
    for i in order:
        d = down[i]
        if d >= 0:
            acc[d] += acc[i]
    return acc.reshape(filled.shape)


def hand(filled: np.ndarray, dem: np.ndarray, down: np.ndarray, drainage: np.ndarray):
    """Return (HAND [m], index of the drainage cell each cell drains to)."""
    flat_f, flat_d = filled.ravel(), dem.ravel()
    ref = np.full(filled.size, -1)
    drain = drainage.ravel()
    order = np.argsort(flat_f, kind="stable")  # downstream (low) first
    for i in order:
        if drain[i]:
            ref[i] = i
        elif down[i] >= 0:
            ref[i] = ref[down[i]]
    # Cells draining off-grid without meeting a channel: reference = their own outlet path minimum (HAND >= 0).
    no_ref = ref < 0
    h = np.where(no_ref, np.nan, flat_d - flat_d[np.maximum(ref, 0)])
    return np.clip(h, 0, None).reshape(filled.shape), ref.reshape(filled.shape)


def zone_links(zone: np.ndarray, down: np.ndarray, drainage: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Downstream zone of each zone, defined through the zone OUTLET (its drainage cell with maximum accumulation).
    Accumulation strictly increases downstream, so the resulting zone graph is acyclic (a river that leaves a
    block and re-enters it cannot create a cycle). -1 = drains out of the grid."""
    Z = int(zone.max()) + 1
    fz, fd, fa, down = zone.ravel(), drainage.ravel(), acc.ravel(), down.ravel()
    dz = -np.ones(Z, int)
    for z in range(Z):
        cells = np.flatnonzero((fz == z) & fd)
        if cells.size == 0:
            cells = np.flatnonzero(fz == z)
        out = cells[np.argmax(fa[cells])]
        d = down[out]
        if d >= 0 and fz[d] >= 0 and fz[d] != z:
            dz[z] = fz[d]
    return dz


def reach_zones(ref: np.ndarray, down: np.ndarray, drainage: np.ndarray, block: int, acc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition the scene into reach zones: drainage cells grouped by `block`-sized spatial blocks,
    every other cell inherits the zone of its drainage reference. Returns (zone map, downstream zone per zone)."""
    H, W = ref.shape
    rr, cc = np.indices((H, W))
    nbc = (W + block - 1) // block
    zone_of_cell = (rr // block) * nbc + (cc // block)
    flat_ref = ref.ravel()
    zone = np.where(flat_ref >= 0, zone_of_cell.ravel()[np.maximum(flat_ref, 0)], -1)
    # relabel to 0..Z-1 among zones that contain drainage cells
    ids = np.unique(zone[zone >= 0])
    lut = -np.ones(zone_of_cell.max() + 1, int); lut[ids] = np.arange(len(ids))
    zone = np.where(zone >= 0, lut[np.maximum(zone, 0)], -1).reshape(H, W)
    return zone, zone_links(zone, down, drainage, acc)


def terrain_products(dem: np.ndarray, channel_cells: int = 200, zone_block: int = 48) -> dict:
    filled = fill_depressions(dem)
    down = flow_directions(filled)
    acc = accumulation(filled, down)
    drainage = acc >= channel_cells
    h, ref = hand(filled, dem, down, drainage)
    zone, dzone = reach_zones(ref, down, drainage, zone_block, acc)
    return {"dem": dem.astype(np.float32), "filled": filled.astype(np.float32), "down": down, "acc": acc.astype(np.float32),
            "drainage": drainage, "hand": h.astype(np.float32), "zone": zone, "zone_down": dzone}
