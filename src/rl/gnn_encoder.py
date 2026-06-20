"""Bidirectional GraphSAGE encoder over the DAG (TZ §6.1)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv


class _BiSAGELayer(nn.Module):
    """One GraphSAGE layer aggregating from predecessors AND successors.

    Direction is flagged by using two separate SAGEConv streams: one over the
    forward edge_index (incoming from predecessors) and one over the reversed
    edge_index (incoming from successors). Their outputs are concatenated and
    projected, so each task embedding sees both directions.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fwd = SAGEConv(in_channels, out_channels)
        self.rev = SAGEConv(in_channels, out_channels)
        self.proj = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        rev_index = edge_index.flip(0) if edge_index.numel() else edge_index
        hf = self.fwd(x, edge_index)
        hr = self.rev(x, rev_index)
        return F.relu(self.proj(torch.cat([hf, hr], dim=-1)))


class GNNEncoder(nn.Module):
    def __init__(
        self, task_in: int = 15, node_in: int = 9, hidden: int = 64, layers: int = 2
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(task_in, hidden)
        self.layers = nn.ModuleList(_BiSAGELayer(hidden, hidden) for _ in range(layers))
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )

    def forward(
        self, task_features: Tensor, edge_index: Tensor, node_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = F.relu(self.input_proj(task_features))
        for layer in self.layers:
            x = layer(x, edge_index)
        h = x
        g = h.mean(dim=0)
        n_emb = self.node_mlp(node_features)
        c = n_emb.mean(dim=0)
        return h, g, n_emb, c
