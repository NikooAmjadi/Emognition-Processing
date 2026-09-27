from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneGroupOut,
    ParameterGrid,
)

from config import Config
from metrics import compute_metrics
from models.specs import ModelSpec
from runners.types import ModelRunResult


def _threshold_used(config: Config) -> float | None:
    if config.classification != "binary":
        return None
    return 0.0 if config.normalize_labels else 5.0


def run_tabular_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: Config,
    input_mode: str,
) -> ModelRunResult:
    pipeline = spec.factory(config)
    param_grid = spec.param_grid_factory(config)
    total_combinations = len(list(ParameterGrid(param_grid)))

    print(f"\nRunning {spec.name} | Mode: {input_mode}")
    print(f"  Number of grid combinations: {total_combinations}")

    fold_metrics: list[dict[str, float]] = []
    subject_rows: list[dict[str, object]] = []
    logo = LeaveOneGroupOut()

    for fold_index, (train_index, test_index) in enumerate(
        logo.split(X, y, groups),
        start=1,
    ):
        subject_id = groups.iloc[test_index].iloc[0]
        print(f"\n  Fold {fold_index} | Subject {subject_id}")

        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        groups_train = groups.iloc[train_index].to_numpy()

        unique_train_groups = np.unique(groups_train)
        inner_splits = min(
            config.inner_cv_splits,
            len(unique_train_groups),
        )
        if inner_splits < 2:
            raise ValueError(
                "Not enough training subjects for inner GroupKFold."
            )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=config.scoring_metric,
            cv=GroupKFold(n_splits=inner_splits),
            n_jobs=config.n_jobs_search,
            verbose=1,
            pre_dispatch="1*n_jobs",
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train, groups=groups_train)

        predictions = search.best_estimator_.predict(X_test)
        metrics = compute_metrics(
            y_test,
            predictions,
            config.classification,
        )
        fold_metrics.append(metrics)

        subject_rows.append(
            {
                "model": spec.name,
                "subject_id": subject_id,
                "threshold_used": _threshold_used(config),
                "neutral_label_removed": config.remove_baseline_neutral,
                "best_params": json.dumps(
                    search.best_params_,
                    sort_keys=True,
                    default=str,
                ),
                **metrics,
            }
        )

        if config.classification == "regression":
            print(
                f"  MAE={metrics['mae']:.4f} | "
                f"RMSE={metrics['rmse']:.4f} | "
                f"R2={metrics['r2']:.4f}"
            )
        else:
            print(
                f"  F1={metrics['f1']:.4f} | "
                f"Acc={metrics['acc']:.4f}"
            )

    return ModelRunResult(
        model_name=spec.name,
        fold_metrics=fold_metrics,
        subject_rows=subject_rows,
    )
