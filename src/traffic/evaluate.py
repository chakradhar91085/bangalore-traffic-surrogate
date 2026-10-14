"""Evaluate every model on the same test rows: MAE/RMSE/R2 (speed), P/R/F1 + confusion matrix (congestion class),
and inference latency. CLI: python -m traffic.evaluate  (writes artifacts/metrics.json)"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, r2_score)

from .baselines import fit_baselines, predict_baseline
from .data import CLASS_NAMES, build, tabular
from .models import make_model
from .train import ART, DEVICE, tensors


def load_dl(name, ds):
    ck = torch.load(ART / f"{name}.pth", map_location=DEVICE)
    cfg = dict(ck["config"])
    kind = cfg.pop("kind")
    m = make_model(kind, len(ds.roads), ds.train.x.shape[-1], adj=ds.adj, hidden=cfg["hidden"], dropout=cfg["dropout"])
    m.load_state_dict(ck["state_dict"])
    return m.to(DEVICE).eval()


@torch.no_grad()
def predict_dl(model, ds, split):
    x, _, _, mask = tensors(getattr(ds, split))
    logits, sp = model(x, mask)
    return torch.softmax(logits[mask], -1).cpu().numpy(), sp[mask].cpu().numpy() * ds.speed_std + ds.speed_mean


def score(y_cls, y_speed, proba, speed):
    pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_cls, pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": accuracy_score(y_cls, pred),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "per_class_f1": dict(zip(CLASS_NAMES, map(float, f))),
        "confusion_matrix": confusion_matrix(y_cls, pred, labels=[0, 1, 2]).tolist(),
        "speed_mae": float(mean_absolute_error(y_speed, speed)),
        "speed_rmse": float(np.sqrt(mean_squared_error(y_speed, speed))),
        "speed_r2": float(r2_score(y_speed, speed)),
    }


def latency_ms(fn, repeats=50):
    fn()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeats * 1000


def dl_latency(model, ds):
    """Time to forecast one whole 16-road network snapshot (batch of 1) on CPU and GPU."""
    out = {}
    for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        m = model.to(dev)
        x, _, _, mask = (t[:1] for t in tensors(ds.test, dev))
        with torch.no_grad():
            out[dev] = latency_ms(lambda: m(x, mask))
    model.to(DEVICE)
    return out


def run(dl_names=("mlp", "gnn")):
    ds = build()
    _, y_cls, y_speed = tabular(ds, "test")
    results, latency = {}, {}
    for name, models in fit_baselines(ds).items():
        proba, sp = predict_baseline(models, ds, "test")
        results[name] = score(y_cls, y_speed, proba, sp)
        X = tabular(ds, "test")[0][:16]  # latency for one snapshot-sized batch of rows
        clf = models[0]
        for m in models:
            if hasattr(m, "n_jobs"):
                m.n_jobs = 1  # time single-process inference, not thread-pool start-up
        latency[name] = {"cpu": latency_ms(lambda: (clf.predict(X), models[1].predict(X)))}
    for name in dl_names:
        model = load_dl(name, ds)
        proba, sp = predict_dl(model, ds, "test")
        results[f"{name.upper()} ({name})"] = score(y_cls, y_speed, proba, sp)
        latency[f"{name.upper()} ({name})"] = dl_latency(model, ds)
    out = {"results": results, "latency_ms": latency, "n_test_rows": int(len(y_cls)),
           "class_names": CLASS_NAMES, "class_balance_test": np.bincount(y_cls, minlength=3).tolist()}
    (ART / "metrics.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    out = run()
    print(f"{'model':22s} acc    mF1    speedMAE  R2")
    for k, v in out["results"].items():
        print(f"{k:22s} {v['accuracy']:.3f}  {v['macro_f1']:.3f}  {v['speed_mae']:6.2f}   {v['speed_r2']:.3f}")
    print(json.dumps(out["latency_ms"], indent=1))
