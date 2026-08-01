from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from config import Config


@dataclass(frozen=True)
class GraphTrainingResult:
    best_epoch: int
    best_validation_loss: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(config: Config) -> torch.device:
    if config.graph_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "graph_device='cuda' was requested, but CUDA is unavailable."
            )
        return torch.device("cuda")

    if config.graph_device == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _criterion(classification: str) -> nn.Module:
    if classification == "regression":
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def _batch_loss(
    model: nn.Module,
    batch: Any,
    criterion: nn.Module,
    classification: str,
) -> torch.Tensor:
    output = model(batch)
    if classification == "regression":
        return criterion(
            output.reshape(-1),
            batch.y.float().reshape(-1),
        )
    return criterion(output, batch.y.long().reshape(-1))


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    criterion: nn.Module,
    classification: str,
    device: torch.device,
) -> float:
    model.train()
    weighted_loss = 0.0
    sample_count = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = _batch_loss(model, batch, criterion, classification)
        loss.backward()
        optimizer.step()

        batch_size = int(batch.num_graphs)
        weighted_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training graph loader is empty.")
    return weighted_loss / sample_count


def evaluate_graph_loss(
    model: nn.Module,
    loader: DataLoader,
    classification: str,
    device: torch.device,
) -> float:
    model.eval()
    criterion = _criterion(classification)
    weighted_loss = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss = _batch_loss(model, batch, criterion, classification)
            batch_size = int(batch.num_graphs)
            weighted_loss += float(loss.item()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Validation graph loader is empty.")
    return weighted_loss / sample_count


def train_graph_model(
    model: nn.Module,
    train_loader: DataLoader,
    classification: str,
    learning_rate: float,
    weight_decay: float,
    config: Config,
    device: torch.device,
    validation_loader: DataLoader | None = None,
    fixed_epochs: int | None = None,
) -> GraphTrainingResult:
    model.to(device)
    criterion = _criterion(classification)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    epochs = fixed_epochs or config.graph_max_epochs
    best_epoch = epochs
    best_validation_loss: float | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            classification,
            device,
        )

        if validation_loader is None:
            if config.graph_verbose and (
                epoch == 1
                or epoch == epochs
                or epoch % 25 == 0
            ):
                print(
                    f"      Epoch {epoch:03d}/{epochs} | "
                    f"train_loss={train_loss:.6f}"
                )
            continue

        validation_loss = evaluate_graph_loss(
            model,
            validation_loader,
            classification,
            device,
        )

        improved = (
            best_validation_loss is None
            or validation_loss
            < best_validation_loss - config.graph_min_delta
        )
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if config.graph_verbose and (
            epoch == 1
            or epoch % 25 == 0
            or stale_epochs >= config.graph_patience
        ):
            print(
                f"      Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={validation_loss:.6f}"
            )

        if stale_epochs >= config.graph_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return GraphTrainingResult(
        best_epoch=max(1, best_epoch),
        best_validation_loss=best_validation_loss,
    )


def predict_graph_model(
    model: nn.Module,
    loader: DataLoader,
    classification: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_values: list[float] = []
    predictions: list[float] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)

            if classification == "regression":
                predicted = output.reshape(-1)
            else:
                predicted = output.argmax(dim=1)

            true_values.extend(
                batch.y.detach().cpu().numpy().reshape(-1).tolist()
            )
            predictions.extend(
                predicted.detach().cpu().numpy().reshape(-1).tolist()
            )

    return np.asarray(true_values), np.asarray(predictions)
