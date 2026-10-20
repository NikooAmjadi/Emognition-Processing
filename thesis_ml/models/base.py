from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    estimator: Any
    parameter_grid: dict[str, list[Any]]
    supports_shap: bool = True
    is_graph: bool = False


def prefixed_grid(grid: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {f"clf__{key}": value for key, value in grid.items()}
