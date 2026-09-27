from __future__ import annotations

import pandas as pd

from config import Config
from data_loader import (
    load_dataset,
    normalize_labels_subjectwise,
    prepare_inputs,
    remove_baseline_neutral,
)
from metrics import build_summary
from model_registry import get_model_specs
from runners.tabular_runner import run_tabular_model


def run_experiment(
    config: Config,
    feature_mode: str,
    input_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataframe = load_dataset(config, feature_mode)

    if config.remove_baseline_neutral:
        dataframe = remove_baseline_neutral(dataframe, config)
    if config.normalize_labels:
        dataframe = normalize_labels_subjectwise(dataframe, config)

    X, y, groups = prepare_inputs(dataframe, input_mode, config)

    if groups.nunique() < 2:
        raise ValueError(
            "LOSO requires data from at least two distinct subjects."
        )

    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"Number of subjects: {groups.nunique()}")
    print(f"\nTarget distribution ({config.classification}):")
    print(y.value_counts(dropna=False).sort_index())
    print(y.value_counts(normalize=True, dropna=False).sort_index())

    model_fold_metrics: dict[str, list[dict[str, float]]] = {}
    subject_rows: list[dict[str, object]] = []

    for spec in get_model_specs(config):
        if spec.family == "tabular":
            result = run_tabular_model(
                spec=spec,
                X=X,
                y=y,
                groups=groups,
                config=config,
                input_mode=input_mode,
            )
        elif spec.family == "graph":
            # Lazy import keeps classical/MLP runs independent from PyG.
            from runners.graph_runner import run_graph_model

            result = run_graph_model(
                spec=spec,
                X=X,
                y=y,
                groups=groups,
                config=config,
                feature_mode=feature_mode,
                input_mode=input_mode,
            )
        else:
            raise ValueError(
                f"Unsupported model family: {spec.family!r}"
            )

        model_fold_metrics[result.model_name] = result.fold_metrics
        subject_rows.extend(result.subject_rows)

    if not model_fold_metrics:
        raise RuntimeError("No models were executed.")
    if not subject_rows:
        raise RuntimeError("No subject-level results were produced.")

    summary_df = build_summary(
        model_fold_metrics,
        config.classification,
    )
    subjects_df = pd.DataFrame(subject_rows)
    return summary_df, subjects_df
