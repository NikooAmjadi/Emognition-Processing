from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ..errors import ExperimentError
from ..tasks import TaskDefinition


@dataclass(frozen=True)
class SubjectFold:
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]


def subject_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def shared_inner_subject_splits(
    reference: pd.DataFrame,
    subject_column: str,
    requested: int,
    seed: int,
    task: TaskDefinition,
) -> tuple[list[SubjectFold], str, int]:
    """Create one subject-level split plan reused by every candidate."""
    groups = reference[subject_column].astype(str).to_numpy()
    y = reference["y"].to_numpy(dtype=int if task.is_classification else float)
    maximum = min(requested, len(np.unique(groups)))
    placeholder = np.zeros((len(reference), 1), dtype=float)
    for count in range(maximum, 1, -1):
        strategies = (
            [
                (
                    "StratifiedGroupKFold",
                    StratifiedGroupKFold(
                        n_splits=count, shuffle=True, random_state=seed
                    ),
                ),
                ("GroupKFold", GroupKFold(n_splits=count)),
            ]
            if task.is_classification
            else [("GroupKFold", GroupKFold(n_splits=count))]
        )
        for name, splitter in strategies:
            try:
                index_splits = list(splitter.split(placeholder, y, groups))
            except ValueError:
                continue
            if task.is_classification and not all(
                np.unique(y[train]).size == task.class_count
                and np.unique(y[validation]).size == task.class_count
                for train, validation in index_splits
            ):
                continue
            folds = [
                SubjectFold(
                    tuple(sorted(map(str, np.unique(groups[train])))),
                    tuple(sorted(map(str, np.unique(groups[validation])))),
                )
                for train, validation in index_splits
            ]
            return folds, name, count
    raise ExperimentError(
        "Could not construct shared grouped inner folds containing all classes"
    )


def valid_inner_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    requested: int,
    seed: int,
    task: TaskDefinition,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    maximum = min(requested, len(np.unique(groups)))
    for count in range(maximum, 1, -1):
        strategies = (
            [
                (
                    "StratifiedGroupKFold",
                    StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed),
                ),
                ("GroupKFold", GroupKFold(n_splits=count)),
            ]
            if task.is_classification
            else [("GroupKFold", GroupKFold(n_splits=count))]
        )
        for name, splitter in strategies:
            try:
                splits = list(splitter.split(X, y, groups))
            except ValueError:
                continue
            if not task.is_classification or all(
                np.unique(y[train]).size == task.class_count
                and np.unique(y[validation]).size == task.class_count
                for train, validation in splits
            ):
                return splits, name, count
    raise ExperimentError("Could not construct a valid grouped inner cross-validation")
