"""Train the U-Net on training events; select checkpoint on validation events; fit temperature on validation.

Usage: python -m floodtask.seg.train --seed 0
Test events are never loaded here.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from floodtask.eval.metrics import scores_from_confusion
from floodtask.seg.dataset import IGNORE, ChipDataset, chips_for, compute_norm_stats
from floodtask.seg.unet import UNet
from floodtask.utils import RESULTS, provenance, save_json, seed_everything


def dice_loss(logits, y):
    valid = (y != IGNORE).float()
    p = torch.softmax(logits, 1)[:, 1] * valid
    t = (y == 1).float() * valid
    return 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)


@torch.no_grad()
def evaluate_loader(model, loader, device, temperature=1.0):
    model.eval()
    tp = fp = fn = tn = 0
    logits_all, y_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        lg = model(x)
        pred = lg.argmax(1)
        v = y != IGNORE
        tp += int(((pred == 1) & (y == 1) & v).sum()); fp += int(((pred == 1) & (y == 0) & v).sum())
        fn += int(((pred == 0) & (y == 1) & v).sum()); tn += int(((pred == 0) & (y == 0) & v).sum())
        # subsample pixels for temperature fitting
        idx = v.nonzero(as_tuple=True)
        keep = torch.randperm(idx[0].numel(), device=device)[:20000]
        logits_all.append(lg.permute(0, 2, 3, 1)[idx[0][keep], idx[1][keep], idx[2][keep]].cpu())
        y_all.append(y[idx[0][keep], idx[1][keep], idx[2][keep]].cpu())
    return scores_from_confusion(tp, fp, fn, tn), torch.cat(logits_all), torch.cat(y_all)


def fit_temperature(logits, y):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp())


def main(seed=0, epochs=80, batch=8, lr=1e-3, base=32):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_c, val_c = chips_for("train"), chips_for("val")
    stats = compute_norm_stats(train_c)
    ds_tr = ChipDataset(train_c, stats, crop=256, augment=True, seed=seed)
    ds_va = ChipDataset(val_c, stats)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch, shuffle=True, generator=g, drop_last=True)
    dl_va = DataLoader(ds_va, 4)

    ys = np.concatenate([v[1].ravel() for v in ds_tr.cache.values()])
    frac_w = (ys == 1).sum() / (ys != IGNORE).sum()
    weight = torch.tensor([0.5 / (1 - frac_w), 0.5 / frac_w], dtype=torch.float32, device=device)

    torch.backends.cudnn.benchmark = True
    model = UNet(base=base).to(device)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    out = RESULTS / "segmentation" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    log, best, t0 = [], -1.0, time.time()
    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                lg = model(x)
            lg = lg.float()  # losses in fp32
            loss = F.cross_entropy(lg, y, weight=weight, ignore_index=IGNORE) + dice_loss(lg, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        val, _, _ = evaluate_loader(model, dl_va, device)
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}})
        if val["iou"] > best:
            best = val["iou"]
            torch.save(model.state_dict(), out / "best.pt")
        print(f"ep {ep:3d} loss {np.mean(losses):.4f} val IoU {val['iou']:.4f} (best {best:.4f})", flush=True)

    model.load_state_dict(torch.load(out / "best.pt", weights_only=True))
    _, lg, y = evaluate_loader(model, dl_va, device)
    T = fit_temperature(lg, y)
    model.temperature.fill_(T)
    torch.save(model.state_dict(), out / "best.pt")
    n_params = sum(p.numel() for p in model.parameters())
    save_json({"seed": seed, "epochs": epochs, "batch": batch, "lr": lr, "base": base, "amp_fp16": device == "cuda", "class_weight": weight.tolist(),
               "train_water_fraction": float(frac_w), "best_val_iou": best, "temperature": T, "n_params": n_params,
               "train_minutes": (time.time() - t0) / 60, "train_events": sorted({c['event'] for c in train_c}),
               "val_events": sorted({c['event'] for c in val_c}), "log": log, "provenance": provenance()},
              out / "train_log.json")
    print(f"done: best val IoU {best:.4f}, T={T:.3f}, params={n_params}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    main(a.seed, a.epochs)
