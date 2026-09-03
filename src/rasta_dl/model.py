"""Graph-temporal neural network used by the RASTA forecasting pipeline."""

import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.projection = nn.Linear(input_features, output_features)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("nm,btmf->btnf", adjacency, values)
        return torch.relu(self.projection(aggregated))


class TrafficSTGNN(nn.Module):
    """Spatial graph convolution followed by node-wise temporal GRU forecasting."""

    def __init__(self, input_features: int = 4, hidden_features: int = 32, horizon: int = 3, outputs: int = 2):
        super().__init__()
        self.horizon = horizon
        self.outputs = outputs
        self.spatial = GraphConv(input_features, hidden_features)
        self.temporal = nn.GRU(hidden_features, hidden_features, batch_first=True)
        self.head = nn.Linear(hidden_features, horizon * outputs)

    def forward(self, values: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial(values, adjacency)
        batch, steps, nodes, hidden = spatial.shape
        temporal_input = spatial.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden)
        encoded, _ = self.temporal(temporal_input)
        forecast = self.head(encoded[:, -1]).reshape(batch, nodes, self.horizon, self.outputs)
        return forecast.permute(0, 2, 1, 3)