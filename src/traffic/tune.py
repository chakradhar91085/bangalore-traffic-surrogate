"""Hyperparameter search (selected on validation loss only), optimizer comparison, and an unregularised
'overfit' reference run. CLI: python -m traffic.tune"""
import itertools

import pandas as pd

from .data import build
from .train import ART, save, train_model

GRID = dict(lr=[1e-3, 3e-3], hidden=[32, 64, 128], dropout=[0.1, 0.3])
OPTIMIZERS = ["adamw", "adam", "rmsprop", "sgd"]


def main():
    ds = build()
    rows = []
    for kind in ("mlp", "gnn"):
        best = (float("inf"), None, None)
        for lr, hidden, dropout in itertools.product(*GRID.values()):
            m, h = train_model(kind, ds, hidden=hidden, dropout=dropout, lr=lr, patience=15, verbose=False)
            v = min(h["val_loss"])
            rows.append(dict(stage="grid", model=kind, lr=lr, hidden=hidden, dropout=dropout, opt="adamw",
                             val_loss=v, val_acc=h["val_acc"][h["best_epoch"] - 1], epochs=h["epochs_run"]))
            print(rows[-1])
            if v < best[0]:
                best = (v, m, h)
        cfg = best[2]["config"]
        for opt in OPTIMIZERS:
            m, h = train_model(kind, ds, hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"], patience=15,
                               verbose=False, opt_name=opt)
            v = min(h["val_loss"])
            rows.append(dict(stage="optimizer", model=kind, lr=cfg["lr"], hidden=cfg["hidden"], dropout=cfg["dropout"],
                             opt=opt, val_loss=v, val_acc=h["val_acc"][h["best_epoch"] - 1], epochs=h["epochs_run"]))
            print(rows[-1])
            if v < best[0]:
                best = (v, m, h)
        save(best[1], best[2], kind)
        # reference: large network, no dropout, no early stopping -> shows overfitting vs the tuned model
        m, h = train_model(kind, ds, hidden=256, dropout=0.0, lr=cfg["lr"], weight_decay=0.0, patience=0,
                           epochs=200, verbose=False)
        save(m, h, f"{kind}_overfit")
        print(kind, "best config:", best[2]["config"])
    ART.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ART / "tuning.csv", index=False)


if __name__ == "__main__":
    main()
