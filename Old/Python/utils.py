from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import Config


def build_output_paths(
    config: Config,
    input_type: str,
    input_mode: str,
) -> tuple[str, str]:
    norm_suffix = (
        "normalized" if config.normalize_labels else "notNormalized"
    )
    filter_suffix = (
        "noBaselineNeutral"
        if config.remove_baseline_neutral
        else "noFilter"
    )
    fast_suffix = "fast" if config.fast_mode else "gridsearch"

    format_values = {
        "target": config.target_label,
        "input_type": input_type,
        "mode": input_mode,
        "class_name": config.classification,
        "norm": norm_suffix,
        "filter": filter_suffix,
        "is_fast": fast_suffix,
    }

    summary_path = config.output_summary_template.format(**format_values)
    subjects_path = config.output_subjects_template.format(**format_values)
    return summary_path, subjects_path


def save_results(
    summary_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    config: Config,
    input_type: str,
    input_mode: str,
) -> None:
    summary_path, subjects_path = build_output_paths(
        config,
        input_type,
        input_mode,
    )

    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(subjects_path).parent.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(summary_path, index=False)
    subjects_df.to_csv(subjects_path, index=False)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved subjects: {subjects_path}")
