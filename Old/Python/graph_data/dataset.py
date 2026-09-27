from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class EmotionGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        graphs: Sequence[Data],
        labels: Sequence[float],
        classification: str,
    ) -> None:
        if len(graphs) != len(labels):
            raise ValueError(
                "The number of graphs and labels must be identical."
            )
        self.graphs = list(graphs)
        self.labels = np.asarray(labels)
        self.classification = classification

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        graph = self.graphs[index].clone()
        if self.classification == "regression":
            graph.y = torch.tensor(
                float(self.labels[index]),
                dtype=torch.float32,
            )
        else:
            graph.y = torch.tensor(
                int(self.labels[index]),
                dtype=torch.long,
            )
        return graph


def create_graph_loader(
    graphs: Sequence[Data],
    labels: Sequence[float],
    classification: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = EmotionGraphDataset(graphs, labels, classification)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
