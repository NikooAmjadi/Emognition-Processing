from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from ..errors import ExperimentError


class GraphEstimator(BaseEstimator):
    """GCN/GAT over aligned sensor nodes built from tabular descriptors.

    A non-negative entry in ``node_groups`` is a source feature index; ``-1``
    is a structurally missing descriptor slot. Each source feature must appear
    exactly once across the configured sensor nodes.
    """

    def __init__(
        self,
        model_name: str,
        node_groups: tuple[tuple[int, ...], ...],
        task_kind: str = "binary",
        class_count: int = 2,
        hidden_channels: int = 32,
        heads: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        min_epochs: int = 5,
        patience: int = 30,
        min_delta: float = 1e-5,
        batch_size: int = 32,
        eval_batch_size: int = 128,
        device: str = "auto",
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.node_groups = node_groups
        self.task_kind = task_kind
        self.class_count = class_count
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.random_state = random_state

    @staticmethod
    def _dependencies() -> tuple[Any, ...]:
        try:
            import torch
            from torch import nn
            from torch.optim import Adam
            from torch_geometric.data import Data
            from torch_geometric.loader import DataLoader
            from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
        except ModuleNotFoundError as exc:
            raise ExperimentError("torch and torch-geometric are required for GCN/GAT") from exc
        return torch, nn, Adam, Data, DataLoader, GATConv, GCNConv, global_mean_pool

    def _device(self, torch: Any) -> Any:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ExperimentError("graph_device='cuda' but CUDA is unavailable")
        use_cuda = self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available())
        return torch.device("cuda" if use_cuda else "cpu")

    def _graphs(self, X: Any, y: Any | None = None) -> list[Any]:
        torch, _, _, Data, _, _, _, _ = self._dependencies()
        values = np.asarray(X, dtype=np.float32)
        if not self.node_groups:
            raise ExperimentError("No graph node groups were configured")
        widths = {len(group) for group in self.node_groups}
        if len(widths) != 1:
            raise ExperimentError("Every graph node must have the same number of features")
        node_count = len(self.node_groups)
        identity = np.eye(node_count, dtype=np.float32)
        edges = [(source, target) for source in range(node_count) for target in range(node_count) if source != target]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        labels = None if y is None else np.asarray(y)
        graphs = []
        for row_index, row in enumerate(values):
            nodes = []
            for node_index, indices in enumerate(self.node_groups):
                descriptors = np.zeros(len(indices), dtype=np.float32)
                presence = np.zeros(len(indices), dtype=np.float32)
                for slot, feature_index in enumerate(indices):
                    if feature_index >= 0:
                        descriptors[slot] = row[feature_index]
                        presence[slot] = 1.0
                nodes.append(
                    np.concatenate(
                        [descriptors, presence, identity[node_index]]
                    )
                )
            kwargs: dict[str, Any] = {
                "x": torch.tensor(np.stack(nodes), dtype=torch.float32),
                "edge_index": edge_index.clone(),
            }
            if labels is not None:
                dtype = torch.float32 if self.task_kind == "regression" else torch.long
                kwargs["y"] = torch.tensor([labels[row_index]], dtype=dtype)
            graphs.append(Data(**kwargs))
        return graphs

    def _make_model(self, in_channels: int) -> Any:
        torch, nn, _, _, _, GATConv, GCNConv, global_mean_pool = self._dependencies()
        hidden = self.hidden_channels
        output_count = 1 if self.task_kind == "regression" else self.class_count
        dropout = self.dropout
        model_name = self.model_name
        heads = self.heads

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if model_name == "gcn":
                    self.conv1 = GCNConv(in_channels, hidden)
                    self.conv2 = GCNConv(hidden, hidden)
                    final_width = hidden
                else:
                    self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=dropout)
                    self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
                    final_width = hidden
                self.norm1 = nn.LayerNorm(hidden * heads if model_name == "gat" else hidden)
                self.norm2 = nn.LayerNorm(final_width)
                self.output = nn.Linear(final_width, output_count)

            def forward(self, batch: Any) -> Any:
                x = self.conv1(batch.x, batch.edge_index)
                x = self.norm1(x)
                x = torch.relu(x)
                x = torch.dropout(x, dropout, self.training)
                x = self.conv2(x, batch.edge_index)
                x = self.norm2(x)
                x = torch.relu(x)
                pooled = global_mean_pool(x, batch.batch)
                return self.output(pooled)

        return Network()

    def fit(self, X: Any, y: Any, groups: Any | None = None) -> "GraphEstimator":
        torch, nn, Adam, _, DataLoader, _, _, _ = self._dependencies()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.device_ = self._device(torch)
        values = np.asarray(X)
        labels = np.asarray(y)
        graphs = self._graphs(values, labels)
        generator = torch.Generator().manual_seed(self.random_state)

        def criterion_for(selected_labels: np.ndarray) -> Any:
            if self.task_kind == "regression":
                return nn.MSELoss()
            counts = np.bincount(selected_labels.astype(int), minlength=self.class_count)
            if np.any(counts == 0):
                raise ExperimentError("Graph training data is missing a configured class")
            weights = len(selected_labels) / (self.class_count * counts.astype(float))
            return nn.CrossEntropyLoss(
                weight=torch.tensor(weights, dtype=torch.float32, device=self.device_)
            )

        train_indices = np.arange(len(labels))
        validation_indices = np.array([], dtype=int)
        if groups is not None and len(np.unique(groups)) >= 3:
            unique_groups = np.unique(np.asarray(groups))
            random = np.random.default_rng(self.random_state)
            random.shuffle(unique_groups)
            validation_count = max(1, int(round(0.2 * len(unique_groups))))
            candidate = np.isin(np.asarray(groups), unique_groups[:validation_count])
            proposed_train = np.flatnonzero(~candidate)
            proposed_validation = np.flatnonzero(candidate)
            class_ok = (
                self.task_kind == "regression"
                or (
                    np.unique(labels[proposed_train]).size == self.class_count
                    and np.unique(labels[proposed_validation]).size == self.class_count
                )
            )
            if class_ok:
                train_indices, validation_indices = proposed_train, proposed_validation

        selection_graphs = [graphs[index] for index in train_indices]
        selection_loader = DataLoader(
            selection_graphs,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        validation_loader = (
            DataLoader(
                [graphs[index] for index in validation_indices],
                batch_size=self.eval_batch_size,
                shuffle=False,
                num_workers=0,
            )
            if len(validation_indices)
            else None
        )
        selection_criterion = criterion_for(labels[train_indices])
        selection_model = self._make_model(int(graphs[0].x.shape[1])).to(self.device_)
        optimizer = Adam(
            selection_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        best_epoch = self.epochs
        best_loss = float("inf")
        stale = 0
        for epoch in range(1, self.epochs + 1):
            selection_model.train()
            for batch in selection_loader:
                batch = batch.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                output = selection_model(batch)
                loss = (
                    selection_criterion(output.reshape(-1), batch.y.reshape(-1))
                    if self.task_kind == "regression"
                    else selection_criterion(output, batch.y.reshape(-1))
                )
                loss.backward()
                optimizer.step()
            if validation_loader is None:
                continue
            selection_model.eval()
            total_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for batch in validation_loader:
                    batch = batch.to(self.device_)
                    output = selection_model(batch)
                    loss = (
                        selection_criterion(output.reshape(-1), batch.y.reshape(-1))
                        if self.task_kind == "regression"
                        else selection_criterion(output, batch.y.reshape(-1))
                    )
                    total_loss += float(loss.item()) * int(batch.num_graphs)
                    total_samples += int(batch.num_graphs)
            validation_loss = total_loss / total_samples
            if epoch >= self.min_epochs and validation_loss < best_loss - self.min_delta:
                best_loss = validation_loss
                best_epoch = epoch
                stale = 0
            elif epoch >= self.min_epochs:
                stale += 1
            if stale >= self.patience:
                break

        # Refit on every training sample for the selected number of epochs.
        torch.manual_seed(self.random_state)
        self.model_ = self._make_model(int(graphs[0].x.shape[1])).to(self.device_)
        full_criterion = criterion_for(labels)
        full_optimizer = Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        full_loader = DataLoader(
            graphs,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
            num_workers=0,
        )
        for _ in range(best_epoch):
            self.model_.train()
            for batch in full_loader:
                batch = batch.to(self.device_)
                full_optimizer.zero_grad(set_to_none=True)
                output = self.model_(batch)
                loss = (
                    full_criterion(output.reshape(-1), batch.y.reshape(-1))
                    if self.task_kind == "regression"
                    else full_criterion(output, batch.y.reshape(-1))
                )
                loss.backward()
                full_optimizer.step()
        self.best_epoch_ = best_epoch
        if self.task_kind != "regression":
            self.classes_ = np.arange(self.class_count)
        return self

    def _raw(self, X: Any) -> np.ndarray:
        torch, _, _, _, DataLoader, _, _, _ = self._dependencies()
        loader = DataLoader(self._graphs(X), batch_size=self.eval_batch_size, shuffle=False, num_workers=0)
        outputs: list[np.ndarray] = []
        self.model_.eval()
        with torch.no_grad():
            for batch in loader:
                outputs.append(self.model_(batch.to(self.device_)).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, X: Any) -> np.ndarray:
        raw = self._raw(X)
        return raw.reshape(-1) if self.task_kind == "regression" else raw.argmax(axis=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task_kind == "regression":
            raise AttributeError("Regression graph models do not expose probabilities")
        raw = self._raw(X)
        shifted = raw - raw.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)


def graph_node_groups(
    features: list[str],
    representation: str,
    node_tokens: tuple[str, ...],
    node_aliases: dict[str, str] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Align HC/NVG/HVG descriptors into one node per physiological signal.

    Slots are keyed by both feature family and descriptor name, so fusion
    representations remain unambiguous. Missing slots are represented by -1;
    GraphEstimator supplies zero plus an explicit presence mask for them.
    """
    if not node_tokens:
        raise ExperimentError("At least one graph node token is required")
    aliases = node_aliases or {}
    allowed_families = set(representation.split("+"))
    mappings: dict[str, dict[str, int]] = {token: {} for token in node_tokens}
    assigned: set[int] = set()

    for index, feature in enumerate(features):
        parts = feature.split("_", 2)
        if len(parts) != 3:
            raise ExperimentError(
                f"Graph feature {feature!r} must follow family_signal_descriptor naming"
            )
        family, raw_sensor, descriptor = parts
        sensor = aliases.get(raw_sensor, raw_sensor)
        if family not in allowed_families:
            raise ExperimentError(
                f"Feature {feature!r} is outside representation {representation!r}"
            )
        if sensor not in mappings:
            raise ExperimentError(
                f"Feature {feature!r} maps to unknown graph sensor {sensor!r}"
            )
        slot = f"{family}::{descriptor}"
        if slot in mappings[sensor]:
            raise ExperimentError(
                f"Duplicate graph descriptor slot {slot!r} for sensor {sensor!r}"
            )
        mappings[sensor][slot] = index
        assigned.add(index)

    if assigned != set(range(len(features))):
        raise ExperimentError("Not every selected feature was assigned to a graph node")
    empty = [token for token, mapping in mappings.items() if not mapping]
    if empty:
        raise ExperimentError(f"Graph sensors have no selected features: {empty}")

    slots = sorted(set().union(*(set(mapping) for mapping in mappings.values())))
    return tuple(
        tuple(mappings[token].get(slot, -1) for slot in slots)
        for token in node_tokens
    )
