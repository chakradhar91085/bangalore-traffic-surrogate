"""Training loop with early stopping and best-validation checkpointing. CLI: python -m traffic.train --model gnn"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import build
from .models import make_model

ART = Path(__file__).resolve().parents[2] / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tensors(s, device=DEVICE):
    return [torch.as_tensor(a).to(device) for a in (s.x, s.cls, s.speed, s.mask)]


def loss_fn(model, x, cls, speed, mask, speed_w=1.0):
    logits, sp = model(x, mask)
    ce = F.cross_entropy(logits[mask], cls[mask])
    hub = F.smooth_l1_loss(sp[mask], speed[mask])
    return ce + speed_w * hub, ce, hub


@torch.no_grad()
def evaluate_loss(model, data):
    model.eval()
    x, cls, speed, mask = data
    total, ce, hub = loss_fn(model, x, cls, speed, mask)
    acc = (model(x, mask)[0].argmax(-1)[mask] == cls[mask]).float().mean().item()
    return total.item(), ce.item(), hub.item(), acc


def train_model(kind, ds, hidden=64, dropout=0.2, lr=1e-3, weight_decay=1e-4, epochs=300,
                patience=25, batch=32, seed=7, verbose=True, opt_name="adamw"):
    """patience<=0 disables early stopping and keeps the last epoch (used to demonstrate overfitting)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = dict(hidden=hidden, dropout=dropout, opt=opt_name)
    model = make_model(kind, len(ds.roads), ds.train.x.shape[-1], adj=ds.adj, **cfg).to(DEVICE)
    opts = {"adamw": lambda p: torch.optim.AdamW(p, lr=lr, weight_decay=weight_decay),
            "adam": lambda p: torch.optim.Adam(p, lr=lr),
            "rmsprop": lambda p: torch.optim.RMSprop(p, lr=lr),
            "sgd": lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9)}
    opt = opts[opt_name](model.parameters())
    tr, va = tensors(ds.train), tensors(ds.val)
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "val_speed_huber")}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(tr[0].shape[0], device=DEVICE)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model, *(t[idx] for t in tr))[0]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        trl, _, _, tra = evaluate_loss(model, tr)
        val, _, vhub, vaa = evaluate_loss(model, va)
        for k, v in zip(hist, (trl, val, tra, vaa, vhub)):
            hist[k].append(v)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"[{kind}] epoch {ep:3d} train {trl:.4f} val {val:.4f} val_acc {vaa:.3f}")
        if patience > 0 and bad >= patience:
            break
    if patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]) + 1)
    hist["epochs_run"] = len(hist["val_loss"])
    hist["seconds"] = round(time.time() - t0, 1)
    return model, {"config": dict(kind=kind, lr=lr, weight_decay=weight_decay, patience=patience, **cfg), **hist}


def save(model, hist, name):
    ART.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": hist["config"]}, ART / f"{name}.pth")
    (ART / f"{name}_history.json").write_text(json.dumps(hist, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "gnn"], default="gnn")
    p.add_argument("--name", default=None)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=25)
    a = p.parse_args()
    ds = build()
    model, hist = train_model(a.model, ds, a.hidden, a.dropout, a.lr, epochs=a.epochs, patience=a.patience)
    save(model, hist, a.name or a.model)
    print(f"device={DEVICE} best_epoch={hist['best_epoch']} epochs={hist['epochs_run']} seconds={hist['seconds']}")


if __name__ == "__main__":
    main()
