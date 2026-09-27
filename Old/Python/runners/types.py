from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelRunResult:
    model_name: str
    fold_metrics: list[dict[str, float]]
    subject_rows: list[dict[str, Any]]
