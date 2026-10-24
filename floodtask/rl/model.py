"""Actor-critic for tile selection: CNN spatial encoder + (LSTM | feed-forward) memory + spatial action head."""
from __future__ import annotations

import torch
import torch.nn as nn

from floodtask.env.tasking_env import GRID_CHANNELS, N_TILES, VEC_FEATURES
from floodtask.hydro.simulator import NT


class ActorCritic(nn.Module):
    def __init__(self, recurrent: bool = True, width: int = 64, hidden: int = 256):
        super().__init__()
        C = len(GRID_CHANNELS)
        self.recurrent, self.hidden = recurrent, hidden
        # GroupNorm keeps spatial features O(1); without it activations shrank ~2.5x per layer and the
        # tile logits were nearly identical (policy could not learn; see docs/troubleshooting.md).
        conv = lambda cin, d: [nn.Conv2d(cin, width, 3, padding=d, dilation=d), nn.GroupNorm(8, width), nn.ReLU()]
        self.enc = nn.Sequential(*conv(C, 1), *conv(width, 1), *conv(width, 2), *conv(width, 4))
        self.vec = nn.Sequential(nn.Linear(len(VEC_FEATURES), 64), nn.ReLU())
        self.pre = nn.Sequential(nn.Linear(2 * width + 64, hidden), nn.ReLU())
        if recurrent:
            self.core = nn.LSTM(hidden, hidden)
        else:
            self.core = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.film = nn.Linear(hidden, 2 * width)
        # action head sees encoded context AND the raw per-tile inputs (skip connection)
        self.head = nn.Sequential(nn.Conv2d(width + C, width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1))
        self.critic = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.orthogonal_(self.head[-1].weight, 0.1)
        nn.init.zeros_(self.head[-1].bias)

    def initial_state(self, n: int, device):
        z = torch.zeros(1, n, self.hidden, device=device)
        return (z, z.clone())

    def forward(self, grid, vec, mask, state, done=None):
        """grid [T,B,C,H,W], vec [T,B,V], mask [T,B,N]; returns logits [T,B,N], value [T,B], new state.
        `done` [T,B] resets the recurrent state before step t when 1."""
        T, B = grid.shape[:2]
        g = grid.flatten(0, 1)
        f = self.enc(g)                                                      # [TB,W,H,W]
        pooled = torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)
        x = self.pre(torch.cat([pooled, self.vec(vec.flatten(0, 1))], 1))   # [TB,hidden]
        if self.recurrent and done is None:   # no resets inside the sequence: one fused cuDNN call
            o, state = self.core(x.view(T, B, -1), state)
            z = o.flatten(0, 1)
        elif self.recurrent:
            x = x.view(T, B, -1)
            outs, (h, c) = [], state
            for t in range(T):
                if done is not None:
                    keep = (1 - done[t]).view(1, B, 1)
                    h, c = h * keep, c * keep
                o, (h, c) = self.core(x[t:t + 1], (h, c))
                outs.append(o)
            z = torch.cat(outs, 0).flatten(0, 1)
            state = (h, c)
        else:
            z = self.core(x)
        gamma, beta = self.film(z).chunk(2, 1)
        f = f * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        logits = self.head(torch.cat([f, g], 1)).flatten(1)                  # [TB,N]
        logits = logits.masked_fill(~mask.flatten(0, 1), -1e9)
        value = self.critic(z).squeeze(-1)
        return logits.view(T, B, N_TILES), value.view(T, B), state
