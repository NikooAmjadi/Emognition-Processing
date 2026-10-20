from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config_types import AppConfig
from ..domain import Dataset
from ..errors import ExperimentError
from ..utils import safe_token, sha256_file
from .registry import register_dataset_adapter, resolve_dataset_adapter


def resolve_input_paths(config: AppConfig, project_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for configured in config.paths.input_files:
        path = configured if configured.is_absolute() else project_dir / configured
        paths.append(path.resolve())
    for pattern in config.paths.input_globs:
        paths.extend(
            path.resolve()
            for path in sorted(project_dir.glob(pattern), key=lambda item: str(item))
        )
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise ExperimentError(
            "No input files matched paths.input_files or paths.input_globs"
        )
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise ExperimentError(f"Input files do not exist: {missing}")
    return unique


def _trial_ids(table: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = table.loc[:, list(columns)].astype(str)
    return values.agg("|".join, axis=1)


def _validate_manifest(path: Path, table: pd.DataFrame) -> tuple[Path | None, str | None]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        return None, None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Invalid manifest {manifest_path}: {exc}") from exc
    output = payload.get("output", {})
    if "csv" in output and output["csv"] != path.name:
        raise ExperimentError(f"Manifest {manifest_path.name} describes another CSV")
    if "row_count" in output and int(output["row_count"]) != len(table):
        raise ExperimentError(f"Manifest row count differs for {path.name}")
    if "column_count" in output and int(output["column_count"]) != len(table.columns):
        raise ExperimentError(f"Manifest column count differs for {path.name}")
    return manifest_path, sha256_file(manifest_path)


def load_csv_dataset(path: Path, config: AppConfig) -> Dataset:
    schema = config.schema
    table = pd.read_csv(path, low_memory=False)
    required = {
        schema.subject_column,
        *schema.trial_id_columns,
        *schema.target_columns.values(),
    }
    if schema.window_id_column:
        required.add(schema.window_id_column)
    missing = required - set(table.columns)
    if missing:
        raise ExperimentError(f"{path.name} is missing columns: {sorted(missing)}")
    if table.empty:
        raise ExperimentError(f"{path.name} contains no rows")
    if table.columns.duplicated().any():
        raise ExperimentError(f"{path.name} contains duplicate column names")

    feature_columns = {
        family: sorted(
            column for column in table.columns if column.startswith(prefix)
        )
        for family, prefix in schema.feature_prefixes.items()
    }
    empty_families = [name for name, columns in feature_columns.items() if not columns]
    if empty_families:
        raise ExperimentError(f"{path.name} has no features for: {empty_families}")
    if schema.expected_feature_counts is not None:
        actual = {name: len(columns) for name, columns in feature_columns.items()}
        if actual != schema.expected_feature_counts:
            raise ExperimentError(
                f"{path.name} feature counts are {actual}; expected "
                f"{schema.expected_feature_counts}"
            )

    features = sum(feature_columns.values(), [])
    numeric = table[features].apply(pd.to_numeric, errors="coerce")
    invalid = numeric.isna() & ~table[features].isna()
    if invalid.to_numpy().any():
        raise ExperimentError(f"{path.name} contains non-numeric feature values")
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ExperimentError(f"{path.name} contains infinite feature values")
    table.loc[:, features] = numeric

    for column in schema.target_columns.values():
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any():
            raise ExperimentError(f"{path.name}: target {column} must be numeric")
        if schema.numeric_target_min is not None and (values < schema.numeric_target_min).any():
            raise ExperimentError(f"{path.name}: {column} is below the configured minimum")
        if schema.numeric_target_max is not None and (values > schema.numeric_target_max).any():
            raise ExperimentError(f"{path.name}: {column} exceeds the configured maximum")
        table[column] = values

    # Upstream manifests describe the complete CSV, before any analysis-specific
    # condition or target filtering is applied.
    manifest_path = manifest_sha = None
    if schema.manifest_validation:
        manifest_path, manifest_sha = _validate_manifest(path, table)

    if schema.condition_column:
        condition = table[schema.condition_column].astype(str).str.strip()
        if schema.condition_casefold:
            condition = condition.str.upper()
            excluded = {value.upper() for value in schema.excluded_conditions}
        else:
            excluded = set(schema.excluded_conditions)
        table[schema.condition_column] = condition
        table = table.loc[~condition.isin(excluded)].copy()
    if table.empty:
        raise ExperimentError(f"Condition filtering removed every row from {path.name}")

    table[schema.subject_column] = table[schema.subject_column].astype(str)
    table["trial_id"] = _trial_ids(table, schema.trial_id_columns)
    if schema.window_id_column:
        duplicate_key = [*schema.trial_id_columns, schema.window_id_column]
        if table.duplicated(duplicate_key).any():
            raise ExperimentError(f"{path.name} contains duplicate sample keys")

    for column in schema.target_columns.values():
        if (table.groupby("trial_id")[column].nunique() > 1).any():
            raise ExperimentError(f"{path.name}: {column} changes within a trial")

    metadata: dict[str, Any] = {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
    }
    for source, destination in (
        ("window_size_sec", "window_sec"),
        ("overlap_sec", "overlap_sec"),
    ):
        if source in table:
            values = pd.to_numeric(table[source], errors="coerce").dropna().unique()
            if len(values) == 1:
                metadata[destination] = float(values[0])

    return Dataset(
        dataset_id=schema.dataset_id,
        input_id=safe_token(path.stem),
        path=path,
        sha256=sha256_file(path),
        table=table,
        feature_columns=feature_columns,
        metadata=metadata,
    )


register_dataset_adapter("generic_csv", load_csv_dataset)


def load_datasets(config: AppConfig, project_dir: Path) -> list[Dataset]:
    loader = resolve_dataset_adapter(config.schema.adapter)
    datasets = [loader(path, config) for path in resolve_input_paths(config, project_dir)]
    input_ids = [dataset.input_id for dataset in datasets]
    if len(input_ids) != len(set(input_ids)):
        raise ExperimentError("Input file stems must be unique")
    return datasets


def _apply_subject_policy(datasets: list[Dataset], config: AppConfig) -> list[dict[str, Any]]:
    subject_column = config.schema.subject_column
    minimum = config.data.min_trials_per_subject
    counts = {
        dataset.input_id: dataset.table.groupby(subject_column)["trial_id"].nunique().to_dict()
        for dataset in datasets
    }
    subjects = sorted(set().union(*(set(values) for values in counts.values())))
    eligible: set[str] = set()
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        observed = {dataset.input_id: int(counts[dataset.input_id].get(subject, 0)) for dataset in datasets}
        minimum_observed = min(observed.values())
        keep = minimum == 0 or minimum_observed >= minimum
        if keep:
            eligible.add(subject)
        row: dict[str, Any] = {
            "subject_id": subject,
            "minimum_required_trials": minimum,
            "minimum_observed_trials": minimum_observed,
            "eligible": keep,
            "exclusion_reason": "" if keep else f"fewer_than_{minimum}_trials",
        }
        row.update({f"trials__{key}": value for key, value in observed.items()})
        rows.append(row)
    if not eligible:
        raise ExperimentError("Subject completeness policy removed every subject")
    for dataset in datasets:
        dataset.table = dataset.table.loc[dataset.table[subject_column].isin(eligible)].copy()
    return rows


def _apply_trial_policy(datasets: list[Dataset], config: AppConfig) -> list[dict[str, Any]]:
    trial_sets = [set(dataset.table["trial_id"]) for dataset in datasets]
    common = set.intersection(*trial_sets)
    target_columns = list(config.schema.target_columns.values())
    reference = (
        datasets[0]
        .table.drop_duplicates("trial_id")
        .set_index("trial_id")[target_columns]
    )
    for dataset in datasets[1:]:
        candidate = (
            dataset.table.drop_duplicates("trial_id").set_index("trial_id")[target_columns]
        )
        shared = sorted(set(reference.index) & set(candidate.index))
        if shared and not np.allclose(
            reference.loc[shared].to_numpy(dtype=float),
            candidate.loc[shared].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise ExperimentError(
                "Target labels disagree between input settings for shared trials: "
                f"{datasets[0].input_id} versus {dataset.input_id}"
            )
    rows: list[dict[str, Any]] = []
    for dataset, available in zip(datasets, trial_sets):
        before = len(dataset.table)
        if config.data.trial_policy == "common":
            dataset.table = dataset.table.loc[dataset.table["trial_id"].isin(common)].copy()
        if dataset.table.empty:
            raise ExperimentError(f"Trial policy removed every row from {dataset.input_id}")
        rows.append(
            {
                "input_id": dataset.input_id,
                "trial_policy": config.data.trial_policy,
                "rows_before": before,
                "rows_after": len(dataset.table),
                "trials_before": len(available),
                "trials_after": dataset.table["trial_id"].nunique(),
                "common_trials": len(common),
            }
        )
    return rows


def prepare_datasets(
    config: AppConfig, project_dir: Path
) -> tuple[list[Dataset], list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = load_datasets(config, project_dir)
    inputs = _apply_trial_policy(datasets, config)
    participants = _apply_subject_policy(datasets, config)
    for row, dataset in zip(inputs, datasets):
        row["rows_after_subject_policy"] = len(dataset.table)
        row["trials_after_subject_policy"] = dataset.table["trial_id"].nunique()
        row["subjects_after_subject_policy"] = dataset.table[
            config.schema.subject_column
        ].nunique()
    return datasets, participants, inputs
