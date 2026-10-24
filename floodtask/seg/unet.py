"""Compact U-Net for 2-band SAR flood segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, p_drop=0.0):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
              nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    if p_drop > 0:
        layers.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 2, n_cls: int = 2, base: int = 32, depth: int = 4, p_drop: float = 0.1):
        super().__init__()
        ch = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_ch, ch[0])] + [_block(ch[i], ch[i + 1], p_drop if i >= depth - 2 else 0)
                                                            for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], n_cls, 1)
        # Temperature for post-hoc calibration (fitted on validation events, never on test).
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)

    @torch.no_grad()
    def predict_proba(self, x):
        """Calibrated water probability [B,H,W]. Input H,W must be divisible by 16."""
        return torch.softmax(self.forward(x) / self.temperature, 1)[:, 1]
