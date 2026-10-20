from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ..errors import ExperimentError
from ..tasks import TaskDefinition


def permutation_rows(
    estimator: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    task: TaskDefinition,
    repeats: int,
    seed: int,
    n_jobs: int,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    result = permutation_importance(
        estimator,
        X,
        y,
        scoring=task.scorer,
        n_repeats=repeats,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return [
        {
            **metadata,
            "feature": feature,
            "source_feature": feature.removesuffix("__mean").removesuffix("__std"),
            "importance_mean": float(mean),
            "importance_std": float(std),
            "repeats": repeats,
        }
        for feature, mean, std in zip(features, result.importances_mean, result.importances_std)
    ]


def _output_slices(values: np.ndarray, task: TaskDefinition) -> list[tuple[str, np.ndarray]]:
    array = np.asarray(values)
    if array.ndim == 2:
        positive = task.positive_index()
        positive = 1 if positive is None else positive
        label = "regression" if not task.is_classification else task.class_names[positive]
        return [(label, array)]
    if array.ndim != 3:
        raise ExperimentError(f"Unsupported SHAP value shape: {array.shape}")
    if task.class_count == 2:
        positive = task.positive_index()
        positive = 1 if positive is None else positive
        return [(task.class_names[positive], array[:, :, positive])]
    return [(name, array[:, :, index]) for index, name in enumerate(task.class_names)]


def shap_rows(
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: list[str],
    task: TaskDefinition,
    background_size: int,
    max_test_samples: int,
    seed: int,
    save_local_values: bool,
    metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ExperimentError("SHAP is enabled but the shap package is not installed") from exc
    random = np.random.default_rng(seed)
    background_index = random.choice(len(X_train), size=min(background_size, len(X_train)), replace=False)
    test_count = len(X_test) if max_test_samples <= 0 else min(max_test_samples, len(X_test))
    test_index = np.arange(len(X_test)) if test_count == len(X_test) else np.sort(random.choice(len(X_test), size=test_count, replace=False))
    background = X_train.iloc[background_index]
    explained = X_test.iloc[test_index]
    if task.is_classification:
        def prediction_function(values: Any) -> np.ndarray:
            try:
                return np.asarray(estimator.predict_proba(values), dtype=float)
            except (AttributeError, NotImplementedError):
                return np.asarray(estimator.decision_function(values), dtype=float)
    else:
        prediction_function = estimator.predict
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(prediction_function, masker, algorithm="permutation", feature_names=features, seed=seed)
    explanation = explainer(explained)
    summary: list[dict[str, Any]] = []
    local: list[dict[str, Any]] = []
    for output_label, values in _output_slices(explanation.values, task):
        mean_abs = np.mean(np.abs(values), axis=0)
        mean_signed = np.mean(values, axis=0)
        order = np.argsort(-mean_abs)
        ranks = np.empty(len(features), dtype=int)
        ranks[order] = np.arange(1, len(features) + 1)
        for index, feature in enumerate(features):
            summary.append(
                {
                    **metadata,
                    "output": output_label,
                    "feature": feature,
                    "source_feature": feature.removesuffix("__mean").removesuffix("__std"),
                    "mean_abs_shap": float(mean_abs[index]),
                    "mean_signed_shap": float(mean_signed[index]),
                    "rank_within_fold": int(ranks[index]),
                    "test_samples_explained": len(explained),
                    "explainer": "permutation",
                }
            )
            if save_local_values:
                for row_position, sample_index in enumerate(test_index):
                    local.append(
                        {
                            **metadata,
                            "output": output_label,
                            "sample_index": int(sample_index),
                            "feature": feature,
                            "feature_value": float(explained.iloc[row_position, index]),
                            "shap_value": float(values[row_position, index]),
                        }
                    )
    return summary, local
