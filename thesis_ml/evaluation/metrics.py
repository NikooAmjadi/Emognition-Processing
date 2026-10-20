from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ..errors import ExperimentError
from ..tasks import TaskDefinition
from .aggregation import subject_macro_primary_score


def prediction_scores(estimator: Any, X: pd.DataFrame, task: TaskDefinition) -> np.ndarray:
    if not task.is_classification:
        return np.asarray(estimator.predict(X), dtype=float)
    if hasattr(estimator, "predict_proba"):
        try:
            return np.asarray(estimator.predict_proba(X), dtype=float)
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X), dtype=float)
        if decision.ndim == 2:
            shifted = decision - decision.max(axis=1, keepdims=True)
            exponent = np.exp(shifted)
            return exponent / exponent.sum(axis=1, keepdims=True)
        return decision
    return np.full((len(X), task.class_count), np.nan)


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    scores: np.ndarray | None,
    task: TaskDefinition,
) -> dict[str, Any]:
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if not task.is_classification:
        rho = spearmanr(true, predicted).statistic if len(true) > 1 else math.nan
        return {
            "n_samples": len(true),
            "mae": float(mean_absolute_error(true, predicted)),
            "rmse": float(mean_squared_error(true, predicted) ** 0.5),
            "r2": float(r2_score(true, predicted)) if len(true) > 1 else math.nan,
            "spearman_rho": float(rho),
        }

    labels = np.arange(task.class_count)
    result: dict[str, Any] = {
        "n_samples": len(true),
        "classes_observed": int(np.unique(true).size),
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": (
            float(balanced_accuracy_score(true, predicted))
            if np.unique(true).size == task.class_count
            else math.nan
        ),
        "macro_precision": float(precision_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "confusion_matrix_json": json.dumps(confusion_matrix(true, predicted, labels=labels).tolist()),
        "roc_auc": math.nan,
        "average_precision": math.nan,
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0
    )
    for index, name in enumerate(task.class_names):
        result[f"support__{name}"] = int((true == index).sum())
        result[f"precision__{name}"] = float(precision[index])
        result[f"recall__{name}"] = float(recall[index])
        result[f"f1__{name}"] = float(f1[index])
    if task.class_count == 2:
        matrix = confusion_matrix(true, predicted, labels=labels)
        result.update(
            {
                "tn": int(matrix[0, 0]),
                "fp": int(matrix[0, 1]),
                "fn": int(matrix[1, 0]),
                "tp": int(matrix[1, 1]),
            }
        )
    if scores is not None and np.isfinite(scores).all() and np.unique(true).size == task.class_count:
        if task.class_count == 2:
            positive = task.positive_index()
            positive = 1 if positive is None else positive
            vector = scores[:, positive] if scores.ndim == 2 else scores
            result["roc_auc"] = float(roc_auc_score(true == positive, vector))
            result["average_precision"] = float(average_precision_score(true == positive, vector))
        elif scores.ndim == 2 and scores.shape[1] == task.class_count:
            result["roc_auc"] = float(roc_auc_score(true, scores, multi_class="ovr", average="macro"))
    return result


def bootstrap_intervals(
    predictions: pd.DataFrame,
    repeats: int,
    seed: int,
    task: TaskDefinition,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, float | int]:
    if repeats <= 0 or predictions.empty:
        return {"bootstrap_repeats": 0}
    subjects = predictions["subject_id"].astype(str).unique()
    random = np.random.default_rng(seed)
    metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "mcc") if task.is_classification else ("mae", "rmse", "r2", "spearman_rho")
    samples: dict[str, list[float]] = {name: [] for name in metric_names}
    primary_samples: list[float] = []
    for repeat_index in range(repeats):
        drawn = random.choice(subjects, size=len(subjects), replace=True)
        parts = []
        for draw_index, subject in enumerate(drawn):
            part = predictions.loc[
                predictions["subject_id"].astype(str) == subject
            ].copy()
            # A repeated bootstrap cluster must remain a separate subject copy.
            part["subject_id"] = f"bootstrap_{repeat_index}_{draw_index}"
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
        scores = None
        if task.is_classification and task.class_count == 2 and "y_score" in frame:
            positive = task.positive_index()
            positive = 1 if positive is None else positive
            vector = frame["y_score"].to_numpy(float)
            scores = np.column_stack([1.0 - vector, vector]) if positive == 1 else np.column_stack([vector, 1.0 - vector])
        metrics = compute_metrics(frame["y_true"], frame["y_pred"], scores, task)
        for name in metric_names:
            value = float(metrics[name])
            if np.isfinite(value):
                samples[name].append(value)
        try:
            primary, _, _ = subject_macro_primary_score(frame, task)
        except ExperimentError:
            primary = math.nan
        if np.isfinite(primary):
            primary_samples.append(primary)
        if progress_callback is not None:
            progress_callback(repeat_index + 1, repeats)
    output: dict[str, float | int] = {"bootstrap_repeats": repeats}
    for name, values in samples.items():
        if values:
            output[f"{name}_ci_low"] = float(np.percentile(values, 2.5))
            output[f"{name}_ci_high"] = float(np.percentile(values, 97.5))
        else:
            output[f"{name}_ci_low"] = math.nan
            output[f"{name}_ci_high"] = math.nan
    output["primary_score_bootstrap_valid_repeats"] = len(primary_samples)
    output["primary_score_ci_low"] = (
        float(np.percentile(primary_samples, 2.5)) if primary_samples else math.nan
    )
    output["primary_score_ci_high"] = (
        float(np.percentile(primary_samples, 97.5)) if primary_samples else math.nan
    )
    return output
