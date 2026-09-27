from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from config import Config


@dataclass(frozen=True)
class GraphSchema:
    node_prefixes: list[str]
    node_columns: list[list[str]]

    @property
    def node_count(self) -> int:
        return len(self.node_columns)

    @property
    def features_per_node(self) -> int:
        return len(self.node_columns[0])


class GraphBuilder:
    def __init__(
        self,
        config: Config,
        feature_mode: str,
        input_mode: str,
    ) -> None:
        self.config = config
        self.feature_mode = feature_mode
        self.input_mode = input_mode
        self.schema: GraphSchema | None = None

    def infer_schema(self, X: pd.DataFrame) -> GraphSchema:
        base_prefixes = self.config.graph_prefixes_for(self.feature_mode)
        prefix_leader = "agg_" if self.input_mode == "aggregated" else ""
        prefixes = [f"{prefix_leader}{prefix}" for prefix in base_prefixes]

        node_columns: list[list[str]] = []
        used_columns: set[str] = set()

        for prefix in prefixes:
            columns = [
                column for column in X.columns if column.startswith(prefix)
            ]
            if not columns:
                raise ValueError(
                    f"No graph features matched node prefix {prefix!r}. "
                    "Check graph_node_prefixes and the CSV column names."
                )

            overlap = used_columns.intersection(columns)
            if overlap:
                raise ValueError(
                    "Graph node prefixes overlap. Columns assigned more than "
                    f"once: {sorted(overlap)}"
                )

            used_columns.update(columns)
            node_columns.append(columns)

        feature_counts = [len(columns) for columns in node_columns]
        if len(set(feature_counts)) != 1:
            details = {
                prefix: count
                for prefix, count in zip(prefixes, feature_counts)
            }
            raise ValueError(
                "All graph nodes must currently have the same number of "
                "features so they can share GCN/GAT layers. Counts: "
                f"{details}. Use aligned VG feature sets or add modality-"
                "specific encoders."
            )

        if len(node_columns) < 2:
            raise ValueError("A graph model requires at least two nodes.")

        self.schema = GraphSchema(
            node_prefixes=prefixes,
            node_columns=node_columns,
        )
        return self.schema

    def build(self, X: pd.DataFrame) -> list[Data]:
        schema = self.schema or self.infer_schema(X)

        missing = [
            column
            for columns in schema.node_columns
            for column in columns
            if column not in X.columns
        ]
        if missing:
            raise ValueError(
                f"Graph input is missing schema columns: {missing}"
            )

        edge_index = self._build_edge_index(schema.node_count)
        graphs: list[Data] = []

        for row_values in X.itertuples(index=False, name=None):
            row = dict(zip(X.columns, row_values))
            node_features = [
                np.asarray(
                    [row[column] for column in columns],
                    dtype=np.float32,
                )
                for columns in schema.node_columns
            ]
            x = torch.from_numpy(np.stack(node_features))
            graphs.append(Data(x=x, edge_index=edge_index.clone()))

        return graphs

    def _build_edge_index(self, node_count: int) -> torch.Tensor:
        sources: list[int] = []
        targets: list[int] = []

        for source in range(node_count):
            for target in range(node_count):
                if (
                    source == target
                    and not self.config.graph_include_self_edges
                ):
                    continue
                sources.append(source)
                targets.append(target)

        return torch.tensor(
            [sources, targets],
            dtype=torch.long,
        )
