from __future__ import annotations

import numpy as np
import pandas as pd

from ..config_types import AppConfig
from ..domain import Dataset
from ..errors import ExperimentError
from ..tasks import TaskDefinition


def representation_columns(
    dataset: Dataset, representation: str, config: AppConfig
) -> list[str]:
    families = config.schema.representations[representation]
    return sum((dataset.feature_columns[family] for family in families), [])


def build_model_table(
    dataset: Dataset,
    target_name: str,
    input_mode: str,
    representation: str,
    task: TaskDefinition,
    config: AppConfig,
) -> tuple[pd.DataFrame, list[str]]:
    schema = config.schema
    target_column = schema.target_columns[target_name]
    raw_features = representation_columns(dataset, representation, config)
    frame = dataset.table.copy()
    encoded = task.encode(frame[target_column])
    frame = frame.loc[encoded.index].copy()
    frame["target_raw"] = frame[target_column]
    frame["y"] = encoded
    if task.is_classification and frame["y"].nunique() != task.class_count:
        raise ExperimentError(
            f"{dataset.input_id}/{target_name}: not all configured classes are present"
        )

    if input_mode == "windowed":
        if schema.window_id_column:
            frame["sample_id"] = (
                frame["trial_id"] + "|w" + frame[schema.window_id_column].astype(str)
            )
        else:
            frame["sample_id"] = frame.index.astype(str)
        return frame, raw_features

    if input_mode != "aggregated":
        raise ExperimentError(f"Unknown input mode: {input_mode}")
    group_columns = [*schema.aggregation_columns, "trial_id"]
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    if (grouped[["target_raw", "y"]].nunique() > 1).any(axis=None):
        raise ExperimentError("Target values are inconsistent within an aggregation unit")
    means = grouped[raw_features].mean().add_suffix("__mean")
    stds = grouped[raw_features].std(ddof=0).add_suffix("__std")
    labels = grouped[["target_raw", "y"]].first()
    aggregated = pd.concat([labels, means, stds], axis=1).reset_index()
    aggregated["sample_id"] = aggregated["trial_id"]
    if schema.window_id_column:
        aggregated[schema.window_id_column] = -1
    features = [
        *(feature + "__mean" for feature in raw_features),
        *(feature + "__std" for feature in raw_features),
    ]
    return aggregated, features
