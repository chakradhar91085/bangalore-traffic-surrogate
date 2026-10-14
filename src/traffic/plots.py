"""Report figures from saved artifacts. CLI: python -m traffic.plots"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .train import ART

FIG = Path(__file__).resolve().parents[2] / "reports" / "figures"


def _hist(name):
    return json.loads((ART / f"{name}_history.json").read_text())


def loss_curves():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for j, kind in enumerate(("mlp", "gnn")):
        for name, label, ls in ((f"{kind}_overfit", "no regularisation (256 units, no dropout, no early stop)", "--"),
                                (kind, "tuned (dropout + early stopping)", "-")):
            h = _hist(name)
            ep = np.arange(1, len(h["val_loss"]) + 1)
            ax[0, j].plot(ep, h["train_loss"], ls, color="C0", label=f"train - {label}" if ls == "-" else None, alpha=.9 if ls == "-" else .5)
            ax[0, j].plot(ep, h["val_loss"], ls, color="C3", label=f"val - {label}" if ls == "-" else None, alpha=.9 if ls == "-" else .5)
            ax[1, j].plot(ep, h["train_acc"], ls, color="C0", alpha=.9 if ls == "-" else .5)
            ax[1, j].plot(ep, h["val_acc"], ls, color="C3", alpha=.9 if ls == "-" else .5)
            if ls == "-":
                ax[0, j].axvline(h["best_epoch"], color="k", lw=.8, ls=":", label=f"best epoch {h['best_epoch']}")
        ax[0, j].set_title(f"{kind.upper()}: loss (dashed = unregularised, solid = tuned)")
        ax[1, j].set_title(f"{kind.upper()}: congestion-class accuracy")
        ax[1, j].set_xlabel("epoch")
        ax[0, j].legend(fontsize=7)
        ax[0, j].set_ylabel("loss (CE + Huber speed)")
        ax[1, j].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png", dpi=150)
    plt.close(fig)


def confusion_matrices(m):
    res = m["results"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for a, (name, r) in zip(axes.flat, res.items()):
        cm = np.array(r["confusion_matrix"])
        a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set_xticks(range(3), m["class_names"])
        a.set_yticks(range(3), m["class_names"])
        a.set_title(f"{name}\nmacro-F1 {r['macro_f1']:.3f}", fontsize=9)
        a.set_xlabel("predicted")
        a.set_ylabel("actual")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def comparison(m):
    res = m["results"]
    names = list(res)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for a, key, title in zip(ax, ("macro_f1", "speed_mae", "speed_r2"),
                             ("Macro F1, congestion class (higher better)", "Speed MAE km/h (lower better)", "Speed R2 (higher better)")):
        v = [res[n][key] for n in names]
        a.barh(names, v, color=["#999"] + ["C0"] * 3 + ["C1"] * 2)
        a.set_title(title, fontsize=10)
        for i, x in enumerate(v):
            a.text(x, i, f" {x:.3f}", va="center", fontsize=8)
        a.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = json.loads((ART / "metrics.json").read_text())
    loss_curves()
    confusion_matrices(m)
    comparison(m)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
