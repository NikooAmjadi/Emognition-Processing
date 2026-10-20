from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_absolute_error

from .config_types import TaskConfig
from .errors import ExperimentError
from .task_presets import TaskPreset, get_task_preset


@dataclass(frozen=True)
class TaskDefinition:
    config: TaskConfig
    preset: TaskPreset

    @property
    def is_classification(self) -> bool:
        return self.config.kind != "regression"

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self.preset.class_groups)

    @property
    def class_count(self) -> int:
        return len(self.class_names) if self.is_classification else 1

    @property
    def scorer(self) -> Any:
        if self.is_classification:
            return make_scorer(balanced_accuracy_score)
        return make_scorer(mean_absolute_error, greater_is_better=False)

    def encode(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        valid = numeric.notna() & ~numeric.isin(self.preset.excluded_values)
        if not self.is_classification:
            return numeric.loc[valid].astype(float)

        encoded = pd.Series(index=numeric.index, dtype="float64")
        for index, (_, members) in enumerate(self.preset.class_groups.items()):
            encoded.loc[numeric.isin(members)] = index
        encoded = encoded.loc[valid & encoded.notna()].astype(int)
        if encoded.empty:
            raise ExperimentError("Target encoding removed every sample")
        return encoded

    def positive_index(self) -> int | None:
        if not self.is_classification or self.class_count != 2:
            return None
        if self.preset.positive_class is None:
            return 1
        try:
            return self.class_names.index(self.preset.positive_class)
        except ValueError as exc:
            raise ExperimentError(
                f"positive_class {self.preset.positive_class!r} is not in class_groups"
            ) from exc


def create_task(config: TaskConfig) -> TaskDefinition:
    return TaskDefinition(config, get_task_preset(config.preset_set, config.kind))
