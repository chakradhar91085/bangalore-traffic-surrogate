"""Evaluate a saved RASTA checkpoint against a chronological test split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import make_synthetic_traffic, split_datasets
from .metrics import regression_metrics
from .model import TrafficSTGNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, default=Path("artifacts/rasta_stgnn.pt"), nargs="?")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    data = make_synthetic_traffic(seed=checkpoint["config"]["seed"])
    _, _, test = split_datasets(data, horizon=checkpoint["config"]["horizon"])
    model = TrafficSTGNN(input_features=checkpoint["config"]["input_features"], horizon=checkpoint["config"]["horizon"])
    model.load_state_dict(checkpoint["model"])
    predictions, targets = [], []
    with torch.no_grad():
        for features, labels in DataLoader(test, batch_size=64):
            predictions.append(model(features, checkpoint["adjacency"]))
            targets.append(labels)
    print(regression_metrics(torch.cat(predictions), torch.cat(targets)))


if __name__ == "__main__":
    main()