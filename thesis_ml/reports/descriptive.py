from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config_types import AppConfig
from ..domain import Dataset
from ..tasks import TaskDefinition
from ..utils import atomic_write_csv


def _distribution_rows(
    dataset: Dataset, target_name: str, task: TaskDefinition, config: AppConfig
) -> list[dict[str, object]]:
    column = config.schema.target_columns[target_name]
    rows: list[dict[str, object]] = []
    for level, raw in (
        ("sample", dataset.table),
        ("trial", dataset.table.drop_duplicates("trial_id")),
    ):
        for value, count in raw[column].value_counts().sort_index().items():
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "input_id": dataset.input_id,
                    "target": target_name,
                    "level": level,
                    "stage": "raw",
                    "label": value,
                    "count": int(count),
                }
            )
        encoded = task.encode(raw[column])
        for value, count in encoded.value_counts().sort_index().items():
            label = task.class_names[int(value)] if task.is_classification else value
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "input_id": dataset.input_id,
                    "target": target_name,
                    "level": level,
                    "stage": task.config.kind,
                    "label": label,
                    "count": int(count),
                }
            )
    return rows


def _correlations(dataset: Dataset) -> pd.DataFrame:
    features = sum(dataset.feature_columns.values(), [])
    correlation = dataset.table[features].corr(method="spearman", min_periods=3)
    family = {feature: feature.split("_", 1)[0] for feature in features}
    rows = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            rho = correlation.at[left, right]
            rows.append(
                {
                    "dataset_id": dataset.dataset_id,
                    "input_id": dataset.input_id,
                    "feature_a": left,
                    "family_a": family[left],
                    "feature_b": right,
                    "family_b": family[right],
                    "same_family": family[left] == family[right],
                    "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
                    "abs_spearman_rho": abs(float(rho)) if pd.notna(rho) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_descriptive_reports(
    run_dir: Path,
    datasets: list[Dataset],
    task: TaskDefinition,
    config: AppConfig,
) -> None:
    if not config.output.save_descriptive_reports:
        return
    if config.analyses.class_distribution:
        rows = [
            row
            for dataset in datasets
            for target in config.task.targets
            for row in _distribution_rows(dataset, target, task, config)
        ]
        atomic_write_csv(pd.DataFrame(rows), run_dir / "class_distribution.csv")
    if config.analyses.correlations:
        correlations = pd.concat([_correlations(dataset) for dataset in datasets], ignore_index=True)
        atomic_write_csv(correlations, run_dir / "feature_correlations_spearman.csv")
        high = correlations.loc[(~correlations["same_family"]) & (correlations["abs_spearman_rho"] >= 0.8)]
        atomic_write_csv(
            high.sort_values("abs_spearman_rho", ascending=False),
            run_dir / "high_cross_family_correlations.csv",
        )
