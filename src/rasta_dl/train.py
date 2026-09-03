"""Train and checkpoint the RASTA graph-temporal forecasting model."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import make_synthetic_traffic, split_datasets
from .metrics import regression_metrics
from .model import TrafficSTGNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, adjacency, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, predictions, targets = 0.0, [], []
    for features, labels in loader:
        output = model(features, adjacency)
        loss = nn.functional.smooth_l1_loss(output, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * len(features)
        predictions.append(output.detach())
        targets.append(labels)
    values = regression_metrics(torch.cat(predictions), torch.cat(targets))
    values["loss"] = total_loss / len(loader.dataset)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rasta_stgnn.pt"))
    args = parser.parse_args()
    seed_everything(args.seed)
    data = make_synthetic_traffic(seed=args.seed)
    train, validation, test = split_datasets(data)
    adjacency = torch.from_numpy(data.adjacency)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)
    model = TrafficSTGNN(horizon=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_validation = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, adjacency, optimizer)
        validation_metrics = run_epoch(model, validation_loader, adjacency)
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(f"epoch={epoch:02d} train_mae={train_metrics['mae']:.3f} validation_mae={validation_metrics['mae']:.3f}")
        if validation_metrics["mae"] < best_validation:
            best_validation = validation_metrics["mae"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "adjacency": adjacency, "config": {"horizon": 3, "input_features": 4, "nodes": len(data.node_names), "seed": args.seed}}, args.output)
    checkpoint = torch.load(args.output, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = run_epoch(model, test_loader, adjacency)
    report = {"test": test_metrics, "best_validation_mae": best_validation, "history": history}
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"test_mae={test_metrics['mae']:.3f} test_rmse={test_metrics['rmse']:.3f}")
    print(f"checkpoint={args.output} report={report_path}")


if __name__ == "__main__":
    main()