from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


class GCNModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = dropout
        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Any) -> torch.Tensor:
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        return self.output_layer(x)


def build_gcn(
    in_channels: int,
    out_channels: int,
    params: dict[str, Any],
) -> GCNModel:
    return GCNModel(
        in_channels=in_channels,
        hidden_channels=int(params["hidden_channels"]),
        out_channels=out_channels,
        dropout=float(params["dropout"]),
    )


def gcn_param_grid(config: Config) -> dict[str, list[Any]]:
    if config.fast_mode:
        return {
            name: [values[0]]
            for name, values in config.gcn_param_grid.items()
        }
    return config.gcn_param_grid


register_model(
    ModelSpec(
        name="GCN",
        family="graph",
        factory=build_gcn,
        param_grid_factory=gcn_param_grid,
    )
)
