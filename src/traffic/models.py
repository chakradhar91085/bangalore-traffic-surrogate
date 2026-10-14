"""Surrogate models over a network snapshot. Input x [B,N,F], mask [B,N] -> (class logits [B,N,3], speed [B,N])."""
import torch
from torch import nn


class Heads(nn.Module):
    def __init__(self, hidden, n_classes=3):
        super().__init__()
        self.cls, self.speed = nn.Linear(hidden, n_classes), nn.Linear(hidden, 1)

    def forward(self, h):
        return self.cls(h), self.speed(h).squeeze(-1)


class NodeMLP(nn.Module):
    """Each road is predicted independently from its own conditions plus a learned road embedding."""

    def __init__(self, n_nodes, n_feat, hidden=64, dropout=0.2, **_):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, 8)
        self.net = nn.Sequential(nn.Linear(n_feat + 8, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.heads = Heads(hidden)

    def forward(self, x, mask):
        e = self.emb.weight.expand(x.shape[0], -1, -1)
        return self.heads(self.net(torch.cat([x, e], -1)))


class GraphNet(nn.Module):
    """Same as NodeMLP but roads exchange information over the road graph (observed neighbours only)."""

    def __init__(self, n_nodes, n_feat, adj, hidden=64, dropout=0.2, layers=2, **_):
        super().__init__()
        self.register_buffer("adj", torch.as_tensor(adj))
        self.emb = nn.Embedding(n_nodes, 8)
        self.inp = nn.Linear(n_feat + 8, hidden)
        self.self_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.nb_w = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.heads = Heads(hidden)

    def forward(self, x, mask):
        e = self.emb.weight.expand(x.shape[0], -1, -1)
        h = self.drop(torch.relu(self.inp(torch.cat([x, e], -1))))
        m = mask.float()
        a = self.adj.unsqueeze(0) * m.unsqueeze(1)
        a = a / a.sum(-1, keepdim=True).clamp(min=1e-6)
        for sw, nw in zip(self.self_w, self.nb_w):
            h = self.drop(torch.relu(sw(h) + nw(a @ (h * m.unsqueeze(-1)))))
        return self.heads(h)


def make_model(kind, n_nodes, n_feat, adj=None, **cfg):
    if kind == "mlp":
        return NodeMLP(n_nodes, n_feat, **cfg)
    if kind == "gnn":
        return GraphNet(n_nodes, n_feat, adj, **cfg)
    raise ValueError(kind)
