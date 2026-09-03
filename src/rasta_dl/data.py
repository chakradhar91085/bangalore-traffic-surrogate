"""Reproducible synthetic traffic data for a Delhi NCR corridor graph."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrafficData:
    features: np.ndarray
    targets: np.ndarray
    adjacency: np.ndarray
    node_names: tuple[str, ...]


NODE_NAMES = (
    "Karol Bagh", "Rajouri Garden", "Punjabi Bagh", "Dhaula Kuan",
    "Mahipalpur", "Aerocity", "Vasant Kunj", "RK Puram",
    "Lajpat Nagar", "ITO", "Kashmere Gate", "Connaught Place",
)


def corridor_adjacency(node_count: int = len(NODE_NAMES)) -> np.ndarray:
    """Build a normalized adjacency matrix for the corridor and shortcuts."""
    adjacency = np.eye(node_count, dtype=np.float32)
    for node in range(node_count):
        adjacency[node, (node - 1) % node_count] = 1
        adjacency[node, (node + 1) % node_count] = 1
    for left, right in ((0, 9), (3, 7), (4, 5), (8, 11)):
        adjacency[left, right] = adjacency[right, left] = 1
    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.sqrt(degree @ degree.T)


def make_synthetic_traffic(steps: int = 960, seed: int = 7) -> TrafficData:
    """Generate speed, flow, rain, and incident features with realistic cycles."""
    rng = np.random.default_rng(seed)
    node_count = len(NODE_NAMES)
    time = np.arange(steps, dtype=np.float32)
    hour = (time % 288) / 12.0
    morning = np.exp(-((hour - 9.0) ** 2) / 5.0)
    evening = np.exp(-((hour - 18.0) ** 2) / 6.0)
    rain = np.clip(rng.gamma(1.1, 0.22, steps) + 0.45 * np.sin(time / 37.0), 0, 1)
    rain[(time.astype(int) % 211) > 196] += 0.55
    rain = np.clip(rain, 0, 1)
    node_bias = np.linspace(-2.8, 2.1, node_count, dtype=np.float32)
    congestion = (morning * 7.5 + evening * 10.0)[..., None] + rain[..., None] * 5.5
    speed = 41.0 - congestion + node_bias + rng.normal(0, 1.1, (steps, node_count))
    flow = 340.0 + 180.0 * (morning + evening)[..., None] + (node_count - node_bias) * 9 + rng.normal(0, 16, (steps, node_count))
    incidents = np.zeros((steps, node_count), dtype=np.float32)
    for start in range(120, steps, 197):
        incidents[start:min(start + 18, steps), (start // 197) % node_count] = 1
    speed -= incidents * 12
    flow += incidents * 40
    features = np.stack((speed, flow, np.broadcast_to(rain[:, None], (steps, node_count)), incidents), axis=-1).astype(np.float32)
    targets = features[..., :2].copy()
    return TrafficData(features, targets, corridor_adjacency(node_count), NODE_NAMES)


class WindowedTrafficDataset(Dataset):
    """Turn a time series into leakage-free lookback and forecast windows."""

    def __init__(self, data: TrafficData, start: int, end: int, lookback: int = 12, horizon: int = 3):
        if start < lookback or end > len(data.features) - horizon or start >= end:
            raise ValueError("Invalid split boundaries for lookback and horizon")
        self.features = torch.from_numpy(data.features)
        self.targets = torch.from_numpy(data.targets)
        self.start = start
        self.end = end
        self.lookback = lookback
        self.horizon = horizon

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        time = self.start + index
        return (self.features[time - self.lookback:time], self.targets[time:time + self.horizon])


def split_datasets(data: TrafficData, lookback: int = 12, horizon: int = 3) -> tuple[WindowedTrafficDataset, WindowedTrafficDataset, WindowedTrafficDataset]:
    """Use chronological train/validation/test splits to prevent future leakage."""
    total = len(data.features)
    train_end, validation_end = int(total * 0.70), int(total * 0.85)
    return (
        WindowedTrafficDataset(data, lookback, train_end, lookback, horizon),
        WindowedTrafficDataset(data, train_end, validation_end, lookback, horizon),
        WindowedTrafficDataset(data, validation_end, total - horizon, lookback, horizon),
    )