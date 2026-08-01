from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import Config


def _read_csv(path: str, dataset_name: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset was not found: {csv_path.resolve()}"
        )
    return pd.read_csv(csv_path)


def combine_raw_vg(
    raw_df: pd.DataFrame,
    vg_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    join_cols = [
        config.subject_col,
        config.emotion_col,
        *config.window_columns,
    ]
    label_cols = list(config.all_label_cols)
    required_cols = join_cols + label_cols

    missing_raw_cols = [
        column for column in required_cols if column not in raw_df.columns
    ]
    missing_vg_cols = [
        column for column in required_cols if column not in vg_df.columns
    ]

    if missing_raw_cols:
        raise ValueError(
            "Required columns are missing from the raw dataset: "
            f"{missing_raw_cols}"
        )
    if missing_vg_cols:
        raise ValueError(
            "Required columns are missing from the VG dataset: "
            f"{missing_vg_cols}"
        )

    raw_duplicate_count = int(raw_df.duplicated(join_cols).sum())
    vg_duplicate_count = int(vg_df.duplicated(join_cols).sum())

    if raw_duplicate_count:
        raise ValueError(
            "Raw dataset contains duplicate merge keys. "
            f"Duplicate rows: {raw_duplicate_count}"
        )
    if vg_duplicate_count:
        raise ValueError(
            "VG dataset contains duplicate merge keys. "
            f"Duplicate rows: {vg_duplicate_count}"
        )

    excluded_cols = set(required_cols)
    raw_feature_cols = [
        column for column in raw_df.columns if column not in excluded_cols
    ]
    vg_feature_cols = [
        column for column in vg_df.columns if column not in excluded_cols
    ]

    if not raw_feature_cols:
        raise ValueError("Raw dataset does not contain feature columns.")
    if not vg_feature_cols:
        raise ValueError("VG dataset does not contain feature columns.")

    raw_part = raw_df[required_cols + raw_feature_cols].copy()
    vg_part = vg_df[required_cols + vg_feature_cols].copy()

    raw_part = raw_part.rename(
        columns={column: f"raw_{column}" for column in raw_feature_cols}
    )
    vg_part = vg_part.rename(
        columns={column: f"vg_{column}" for column in vg_feature_cols}
    )

    combined_df = raw_part.merge(
        vg_part,
        on=join_cols,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_raw", "_vg"),
        sort=False,
    )

    if not (combined_df["_merge"] == "both").all():
        raw_only_count = int(
            (combined_df["_merge"] == "left_only").sum()
        )
        vg_only_count = int(
            (combined_df["_merge"] == "right_only").sum()
        )
        raise ValueError(
            "Raw and VG datasets are not fully aligned. "
            f"Raw-only rows: {raw_only_count}, "
            f"VG-only rows: {vg_only_count}"
        )

    for label_col in label_cols:
        raw_label_col = f"{label_col}_raw"
        vg_label_col = f"{label_col}_vg"

        raw_labels = pd.to_numeric(
            combined_df[raw_label_col], errors="coerce"
        ).to_numpy(dtype=float)
        vg_labels = pd.to_numeric(
            combined_df[vg_label_col], errors="coerce"
        ).to_numpy(dtype=float)

        matching_labels = np.isclose(
            raw_labels,
            vg_labels,
            equal_nan=True,
        )
        if not matching_labels.all():
            mismatch_count = int((~matching_labels).sum())
            raise ValueError(
                f"Label {label_col!r} differs between raw and VG datasets "
                f"in {mismatch_count} rows."
            )

        combined_df[label_col] = combined_df[raw_label_col]
        combined_df = combined_df.drop(
            columns=[raw_label_col, vg_label_col]
        )

    combined_df = combined_df.drop(columns="_merge")

    output_columns = (
        join_cols
        + label_cols
        + [f"raw_{column}" for column in raw_feature_cols]
        + [f"vg_{column}" for column in vg_feature_cols]
    )
    return combined_df[output_columns]


def load_dataset(config: Config, input_type: str) -> pd.DataFrame:
    if input_type == "raw":
        return _read_csv(config.raw_dataset_path, "Raw")

    if input_type == "vg":
        return _read_csv(config.vg_dataset_path, "VG")

    if input_type == "rawvg":
        raw_df = _read_csv(config.raw_dataset_path, "Raw")
        vg_df = _read_csv(config.vg_dataset_path, "VG")
        return combine_raw_vg(raw_df, vg_df, config)

    raise ValueError("input_type must be 'raw', 'vg', or 'rawvg'.")


def remove_baseline_neutral(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    if config.emotion_col not in df.columns:
        raise ValueError(
            f"Emotion column {config.emotion_col!r} was not found."
        )

    filtered = df.copy()
    original_count = len(filtered)
    filtered = filtered[
        ~filtered[config.emotion_col]
        .astype(str)
        .str.upper()
        .isin(["BASELINE", "NEUTRAL"])
    ].copy()

    removed = original_count - len(filtered)
    print(
        f"Removed {removed} baseline/neutral samples. "
        f"Remaining: {len(filtered)}"
    )
    return filtered


def normalize_labels_subjectwise(
    df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    if config.subject_col not in df.columns:
        raise ValueError(
            f"Subject column {config.subject_col!r} was not found."
        )

    normalized = df.copy()
    available = [
        column
        for column in config.all_label_cols
        if column in normalized.columns
    ]
    if not available:
        raise ValueError("No label columns were found for normalization.")

    for column in available:
        normalized[column] = normalized.groupby(config.subject_col)[
            column
        ].transform(
            lambda values: (
                (values - values.mean()) / values.std()
                if values.std() > 0
                else 0.0
            )
        )

    print(f"Subject-wise Z-score applied to: {available}")
    return normalized


def aggregate_features(
    df: pd.DataFrame,
    prefix: str,
    config: Config,
) -> pd.DataFrame:
    grouped = df.copy()
    group_cols = list(config.group_cols)

    missing_group_cols = [
        column for column in group_cols if column not in grouped.columns
    ]
    if missing_group_cols:
        raise ValueError(
            "Aggregation requires metadata columns that are missing: "
            f"{missing_group_cols}"
        )

    excluded_cols = set(
        group_cols + config.window_columns + config.all_label_cols
    )
    feature_cols = [
        column for column in grouped.columns if column not in excluded_cols
    ]
    feature_cols = (
        grouped[feature_cols]
        .select_dtypes(include=[np.number])
        .columns.tolist()
    )

    if not feature_cols:
        raise ValueError(
            "No numeric feature columns were found for aggregation."
        )

    aggregated = (
        grouped.groupby(group_cols, dropna=False, sort=False)[feature_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    flattened_columns: list[str] = []
    for column in aggregated.columns.to_flat_index():
        if isinstance(column, tuple):
            base_name = str(column[0])
            statistic = str(column[1]) if len(column) > 1 else ""
        else:
            base_name = str(column)
            statistic = ""

        if base_name in group_cols and not statistic:
            flattened_columns.append(base_name)
        elif statistic:
            flattened_columns.append(
                f"{prefix}_{base_name}_{statistic}"
            )
        else:
            flattened_columns.append(f"{prefix}_{base_name}")

    aggregated.columns = flattened_columns
    return aggregated


def prepare_inputs(
    df: pd.DataFrame,
    input_mode: str,
    config: Config,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if input_mode not in {"windowed", "aggregated"}:
        raise ValueError("input_mode must be 'windowed' or 'aggregated'.")
    if config.label_col not in df.columns:
        raise ValueError(
            f"Label column {config.label_col!r} was not found."
        )
    if config.subject_col not in df.columns:
        raise ValueError(
            f"Subject column {config.subject_col!r} was not found."
        )

    prepared = df.copy()
    if input_mode == "aggregated":
        prepared = aggregate_features(prepared, "agg", config)

    drop_cols = (
        [config.subject_col, config.emotion_col]
        + config.all_label_cols
        + config.window_columns
    )
    drop_cols = [
        column for column in drop_cols if column in prepared.columns
    ]

    X = prepared.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]).astype(np.float32)
    if X.empty or X.shape[1] == 0:
        raise ValueError("No numeric model features remain after preparation.")

    labels = pd.to_numeric(
        prepared[config.label_col], errors="coerce"
    ).astype(float)
    if labels.isna().any():
        raise ValueError(
            f"Label column {config.label_col!r} contains missing or "
            "non-numeric values."
        )

    if config.classification == "binary":
        threshold = 0.0 if config.normalize_labels else 5.0
        y = (labels > threshold).astype(int)
    elif config.classification == "ternary":
        if config.normalize_labels:
            y = pd.cut(
                labels,
                bins=[-np.inf, -1.0, 1.0, np.inf],
                labels=[0, 1, 2],
                include_lowest=True,
            ).astype(int)
        else:
            valid = labels.between(1, 9, inclusive="both")
            if not valid.all():
                invalid_values = sorted(labels[~valid].unique().tolist())
                raise ValueError(
                    "Non-normalized ternary labels must be between 1 and 9. "
                    f"Invalid values: {invalid_values}"
                )
            y = pd.cut(
                labels,
                bins=[0, 3, 6, 9],
                labels=[0, 1, 2],
                include_lowest=True,
            ).astype(int)
    elif config.classification == "regression":
        y = labels.astype(float)
    else:
        raise ValueError(
            "classification must be binary, ternary, or regression."
        )

    groups = prepared[config.subject_col].copy()
    if groups.isna().any():
        raise ValueError("Subject groups contain missing values.")

    # Reset all indices together so positional indexing stays reliable in LOSO.
    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        groups.reset_index(drop=True),
    )
