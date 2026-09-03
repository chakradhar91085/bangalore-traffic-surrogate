"""Metrics for traffic forecasts."""

import torch


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    error = predictions - targets
    return {
        "mae": torch.abs(error).mean().item(),
        "rmse": torch.sqrt(torch.square(error).mean()).item(),
    }