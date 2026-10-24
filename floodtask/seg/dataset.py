"""Sen1Floods11 chip dataset with explicit nodata handling and train-only normalization."""
from __future__ import annotations

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from floodtask.utils import DATA, load_json, load_yaml, ROOT, save_json

RAW = DATA / "raw" / "sen1floods11"
IGNORE = -1
S1_CLIP = (-50.0, 5.0)  # dB; values outside are sensor artefacts / border fill


def chips_for(split: str) -> list[dict]:
    events = set(load_yaml(ROOT / "configs" / "data.yaml")["split"][split])
    return [c for c in load_json(RAW / "chips.json") if c["event"] in events]


def read_chip(chip: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (s1 [2,H,W] dB float32 with NaN nodata, label [H,W] int64 {-1,0,1}, otsu label)."""
    with rasterio.open(RAW / "S1Hand" / f"{chip}_S1Hand.tif") as src:
        s1 = src.read().astype(np.float32)
    with rasterio.open(RAW / "LabelHand" / f"{chip}_LabelHand.tif") as src:
        lab = src.read(1).astype(np.int64)
    with rasterio.open(RAW / "S1OtsuLabelHand" / f"{chip}_S1OtsuLabelHand.tif") as src:
        otsu = src.read(1).astype(np.int64)
    s1[~np.isfinite(s1)] = np.nan
    s1[(s1 < S1_CLIP[0]) | (s1 > S1_CLIP[1])] = np.nan
    lab[lab < 0] = IGNORE
    return s1, lab, otsu


def compute_norm_stats(train_chips: list[dict]) -> dict:
    """Per-band mean/std over valid pixels of TRAINING chips only."""
    sums, sq, n = np.zeros(2), np.zeros(2), np.zeros(2)
    for c in train_chips:
        s1, _, _ = read_chip(c["chip"])
        for b in range(2):
            v = s1[b][np.isfinite(s1[b])].astype(np.float64)
            sums[b] += v.sum()
            sq[b] += (v**2).sum()
            n[b] += v.size
    mean = sums / n
    std = np.sqrt(sq / n - mean**2)
    stats = {"mean": mean.tolist(), "std": std.tolist(), "n_pixels": n.tolist(),
             "chips": [c["chip"] for c in train_chips]}
    save_json(stats, DATA / "processed" / "s1_norm_stats.json")
    return stats


def normalize(s1: np.ndarray, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized image (nodata -> 0) and validity mask."""
    valid = np.isfinite(s1).all(0)
    x = (s1 - np.array(stats["mean"], np.float32)[:, None, None]) / np.array(stats["std"], np.float32)[:, None, None]
    x = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
    return x, valid


class ChipDataset(Dataset):
    def __init__(self, chips: list[dict], stats: dict, crop: int | None = None, augment: bool = False, seed: int = 0):
        self.chips, self.stats, self.crop, self.augment = chips, stats, crop, augment
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        for c in chips:
            s1, lab, _ = read_chip(c["chip"])
            x, valid = normalize(s1, stats)
            lab = lab.copy()
            lab[~valid] = IGNORE  # no supervision where SAR is missing
            self.cache[c["chip"]] = (x, lab)

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, i):
        x, y = self.cache[self.chips[i]["chip"]]
        if self.crop:
            H, W = y.shape
            r, c = self.rng.integers(0, H - self.crop + 1), self.rng.integers(0, W - self.crop + 1)
            x, y = x[:, r:r + self.crop, c:c + self.crop], y[r:r + self.crop, c:c + self.crop]
        if self.augment:
            k = self.rng.integers(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if self.rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
