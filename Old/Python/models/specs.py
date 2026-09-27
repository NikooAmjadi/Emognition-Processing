from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from config import Config


ModelFamily = Literal["tabular", "graph"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: ModelFamily
    factory: Callable[..., Any]
    param_grid_factory: Callable[[Config], dict[str, list[Any]]]
