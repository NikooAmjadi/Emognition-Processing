from __future__ import annotations

import importlib
from pathlib import Path
from typing import Callable

from ..config_types import AppConfig
from ..domain import Dataset
from ..errors import ExperimentError

DatasetLoader = Callable[[Path, AppConfig], Dataset]
_ADAPTERS: dict[str, DatasetLoader] = {}


def register_dataset_adapter(name: str, loader: DatasetLoader) -> None:
    if not name.strip():
        raise ExperimentError("Dataset adapter name cannot be empty")
    _ADAPTERS[name] = loader


def resolve_dataset_adapter(name: str) -> DatasetLoader:
    if name in _ADAPTERS:
        return _ADAPTERS[name]
    if ":" in name:
        module_name, function_name = name.split(":", 1)
        try:
            loader = getattr(importlib.import_module(module_name), function_name)
        except (ImportError, AttributeError) as exc:
            raise ExperimentError(f"Cannot import dataset adapter {name!r}: {exc}") from exc
        if not callable(loader):
            raise ExperimentError(f"Dataset adapter {name!r} is not callable")
        return loader
    raise ExperimentError(
        f"Unknown dataset adapter {name!r}; use a registered name or 'module:function'"
    )
