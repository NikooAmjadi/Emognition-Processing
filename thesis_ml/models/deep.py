from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ..errors import ExperimentError


def _torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.optim import Adam
    except ModuleNotFoundError as exc:
        raise ExperimentError("PyTorch is required for deep-learning models") from exc
    return torch, nn, Adam


class TorchDenseEstimator(BaseEstimator):
    """Small sklearn-compatible PyTorch MLP for classification or regression."""

    def __init__(
        self,
        task_kind: str = "binary",
        class_count: int = 2,
        hidden_channels: int = 128,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int = 32,
        device: str = "cpu",
        random_state: int = 42,
    ):
        self.task_kind = task_kind
        self.class_count = class_count
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "TorchDenseEstimator":
        torch, nn, Adam = _torch()
        torch.manual_seed(self.random_state)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ExperimentError("CUDA was requested but is unavailable")
        device = torch.device(self.device)
        values = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y)
        output_count = 1 if self.task_kind == "regression" else self.class_count
        self.model_ = nn.Sequential(
            nn.Linear(values.shape[1], self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_channels // 2, output_count),
        ).to(device)
        features_tensor = torch.tensor(values, dtype=torch.float32)
        label_dtype = torch.float32 if self.task_kind == "regression" else torch.long
        labels_tensor = torch.tensor(labels, dtype=label_dtype)
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        generator = torch.Generator().manual_seed(self.random_state)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)
        if self.task_kind == "regression":
            criterion = nn.MSELoss()
        else:
            counts = np.bincount(labels.astype(int), minlength=self.class_count)
            weights = len(labels) / (self.class_count * np.maximum(counts, 1))
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
            self.classes_ = np.arange(self.class_count)
        optimizer = Adam(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = self.model_(batch_x)
                loss = criterion(output.reshape(-1), batch_y) if self.task_kind == "regression" else criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        self.device_ = device
        return self

    def _raw(self, X: Any) -> np.ndarray:
        torch, _, _ = _torch()
        self.model_.eval()
        with torch.no_grad():
            values = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device_)
            return self.model_(values).cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        raw = self._raw(X)
        return raw.reshape(-1) if self.task_kind == "regression" else raw.argmax(axis=1)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.task_kind == "regression":
            raise AttributeError("Regression models do not expose predict_proba")
        raw = self._raw(X)
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
