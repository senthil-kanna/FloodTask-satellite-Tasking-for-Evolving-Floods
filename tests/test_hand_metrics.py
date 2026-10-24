import numpy as np

from floodtask.eval.metrics import boundary_f1, map_scores
from floodtask.hydro.hand import accumulation, fill_depressions, flow_directions, terrain_products


def valley(n=64):
    """V-shaped valley draining south: channel along the centre column."""
    r, c = np.indices((n, n))
    return (np.abs(c - n // 2) * 1.0 + (n - r) * 0.1).astype(np.float32)


def test_fill_removes_pits():
    dem = valley()
    dem[30, 20] -= 50  # artificial pit
    filled = fill_depressions(dem)
    down = flow_directions(filled)
    # every interior cell has a downstream neighbour after filling
    assert (down.reshape(dem.shape)[1:-1, 1:-1] >= 0).all()
    assert filled[30, 20] >= dem[30, 20]


def test_accumulation_conserves_cells():
    dem = valley()
    filled = fill_depressions(dem)
    down = flow_directions(filled)
    acc = accumulation(filled, down)
    # total of outlet accumulations equals cell count
    outlets = down == -1
    assert np.isclose(acc.ravel()[outlets].sum(), dem.size)
    assert acc[-1, 32] == acc.max()


def test_hand_zero_on_channel_and_increases_away():
    t = terrain_products(valley(), channel_cells=50, zone_block=16)
    h = t["hand"]
    assert np.all(h[t["drainage"]] == 0)
    row = h[20]
    assert row[32] == 0 and row[40] > row[34] > 0 and row[24] > row[30] > 0


def test_metrics_known_values():
    truth = np.zeros((10, 10), bool); truth[:5] = True
    pred = np.zeros((10, 10), bool); pred[:4] = True; pred[9, 0] = True
    s = map_scores(pred, truth)
    assert (s["tp"], s["fp"], s["fn"]) == (40, 1, 10)
    assert np.isclose(s["iou"], 40 / 51)
    assert np.isclose(s["f1"], 80 / 91)
    assert np.isclose(s["area_bias"], (41 - 50) / 50)
    assert map_scores(np.zeros((3, 3)), np.zeros((3, 3)))["iou"] == 1.0
    assert boundary_f1(truth, truth) == 1.0
    assert boundary_f1(pred, np.zeros_like(truth)) == 0.0


def test_zone_graph_acyclic_on_meandering_river():
    """A river crossing block boundaries back and forth must not create zone cycles."""
    from floodtask.hydro.simulator import _topo_order
    n = 96
    r, c = np.indices((n, n))
    centre = n // 2 + (12 * np.sin(r / 6.0)).astype(int)   # meander crosses the 16-cell block boundaries repeatedly
    dem = (np.abs(c - centre) * 1.0 + (n - r) * 0.05).astype(np.float32)
    t = terrain_products(dem, channel_cells=40, zone_block=16)
    dz = t["zone_down"]
    for z in range(len(dz)):  # follow links: must terminate within Z steps
        seen, cur = set(), z
        while cur >= 0:
            assert cur not in seen
            seen.add(cur); cur = dz[cur]
    assert len(_topo_order(dz)) == len(dz)
