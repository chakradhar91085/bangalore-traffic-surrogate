import numpy as np
import pytest
import torch

from traffic.data import build
from traffic.graph import build_adjacency
from traffic.models import make_model


@pytest.fixture(scope="module")
def ds():
    return build()


def test_chronological_split_no_leakage(ds):
    assert ds.train.dates.max() < ds.val.dates.min() <= ds.val.dates.max() < ds.test.dates.min()
    n = len(ds.train.dates) + len(ds.val.dates) + len(ds.test.dates)
    assert abs(len(ds.train.dates) / n - 0.8) < 0.01


def test_every_row_used_once(ds):
    total = sum(int(s.mask.sum()) for s in (ds.train, ds.val, ds.test))
    assert total == len(ds.frame)


def test_scaler_fit_on_train_only(ds):
    tr = ds.train
    num = tr.x[tr.mask][:, :6]
    assert np.allclose(num.mean(0), 0, atol=1e-4) and np.allclose(num.std(0), 1, atol=1e-2)


def test_adjacency_symmetric_normalised(ds):
    raw, norm = build_adjacency(ds.areas)
    assert np.allclose(raw, raw.T) and np.allclose(norm, norm.T)
    assert np.all(np.diag(raw) == 1)


@pytest.mark.parametrize("kind", ["mlp", "gnn"])
def test_model_shapes_and_finite(ds, kind):
    m = make_model(kind, len(ds.roads), ds.train.x.shape[-1], adj=ds.adj)
    x, mask = torch.as_tensor(ds.test.x[:4]), torch.as_tensor(ds.test.mask[:4])
    logits, sp = m(x, mask)
    assert logits.shape == (4, 16, 3) and sp.shape == (4, 16)
    assert torch.isfinite(logits).all() and torch.isfinite(sp).all()


def test_gnn_ignores_unobserved_neighbours(ds):
    m = make_model("gnn", len(ds.roads), ds.train.x.shape[-1], adj=ds.adj).eval()
    x, mask = torch.as_tensor(ds.test.x[:1]).clone(), torch.as_tensor(ds.test.mask[:1]).clone()
    j = int((~mask[0]).nonzero()[0])
    i = int(mask[0].nonzero()[0])
    a = m(x, mask)[0][0, i]
    x[0, j] = 99.0
    b = m(x, mask)[0][0, i]
    assert torch.allclose(a, b)
