import torch

from rasta_dl.data import make_synthetic_traffic, split_datasets
from rasta_dl.model import TrafficSTGNN


def test_graph_forecast_shape_and_finite_values():
    data = make_synthetic_traffic(steps=120)
    train, _, _ = split_datasets(data, lookback=6, horizon=2)
    features, targets = train[0]
    model = TrafficSTGNN(horizon=2)
    prediction = model(features.unsqueeze(0), torch.from_numpy(data.adjacency))
    assert prediction.shape == targets.unsqueeze(0).shape
    assert torch.isfinite(prediction).all()


def test_chronological_splits_do_not_overlap():
    data = make_synthetic_traffic(steps=200)
    train, validation, test = split_datasets(data)
    assert train.start < train.end <= validation.start < validation.end <= test.start < test.end