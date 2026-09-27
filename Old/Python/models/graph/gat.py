from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


class GATModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout
        self.output_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Any) -> torch.Tensor:
        x = self.gat1(data.x, data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, data.edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, data.batch)
        return self.output_layer(x)


def build_gat(
    in_channels: int,
    out_channels: int,
    params: dict[str, Any],
) -> GATModel:
    return GATModel(
        in_channels=in_channels,
        hidden_channels=int(params["hidden_channels"]),
        out_channels=out_channels,
        heads=int(params["heads"]),
        dropout=float(params["dropout"]),
    )


def gat_param_grid(config: Config) -> dict[str, list[Any]]:
    if config.fast_mode:
        return {
            name: [values[0]]
            for name, values in config.gat_param_grid.items()
        }
    return config.gat_param_grid


register_model(
    ModelSpec(
        name="GAT",
        family="graph",
        factory=build_gat,
        param_grid_factory=gat_param_grid,
    )
)
