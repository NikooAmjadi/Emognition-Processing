from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ExperimentError
from ..tasks import TaskDefinition


def aggregate_predictions_to_trial(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    task: TaskDefinition,
    subject_column: str,
) -> pd.DataFrame:
    """Convert candidate predictions to the common independent trial unit."""
    if not (len(frame) == len(y_true) == len(y_pred) == len(scores)):
        raise ExperimentError("Prediction arrays have incompatible lengths")
    rows = pd.DataFrame(
        {
            "subject_id": frame[subject_column].astype(str).to_numpy(),
            "trial_id": frame["trial_id"].astype(str).to_numpy(),
            "target_raw": frame["target_raw"].to_numpy(),
            "y_true": y_true,
            "y_pred_sample": y_pred,
        }
    )
    score_array = np.asarray(scores)
    if score_array.ndim == 1:
        rows["score_0"] = score_array
    else:
        for index in range(score_array.shape[1]):
            rows[f"score_{index}"] = score_array[:, index]

    output: list[dict[str, Any]] = []
    for (subject, trial), group in rows.groupby(
        ["subject_id", "trial_id"], sort=False, dropna=False
    ):
        if group["y_true"].nunique() != 1:
            raise ExperimentError(f"True target changes within trial {trial}")
        true_value = group["y_true"].iloc[0]
        record: dict[str, Any] = {
            "subject_id": subject,
            "trial_id": trial,
            "sample_id": trial,
            "target_raw": group["target_raw"].iloc[0],
            "y_true": true_value,
            "source_sample_count": len(group),
        }
        score_columns = [column for column in group if column.startswith("score_")]
        if not task.is_classification:
            prediction = float(group["score_0"].mean())
            record.update({"y_pred": prediction, "y_score": prediction})
        elif score_array.ndim == 1:
            mean_score = float(group["score_0"].mean())
            prediction = int(mean_score >= 0.0)
            record.update({"y_pred": prediction, "y_score": mean_score})
        elif score_columns:
            mean_scores = group[score_columns].mean().to_numpy(dtype=float)
            prediction = int(np.argmax(mean_scores))
            record["y_pred"] = prediction
            if task.class_count == 2:
                positive = task.positive_index()
                positive = 1 if positive is None else positive
                record["y_score"] = float(mean_scores[positive])
            else:
                record["y_scores_json"] = json.dumps(mean_scores.tolist())
        else:
            counts = group["y_pred_sample"].value_counts()
            record["y_pred"] = int(counts.index.min() if len(counts) > 1 and counts.iloc[0] == counts.iloc[1] else counts.index[0])
        output.append(record)
    return pd.DataFrame(output)


def subject_macro_primary_score(
    predictions: pd.DataFrame, task: TaskDefinition
) -> tuple[float, int, int]:
    """Calculate the primary score while retaining every observed subject.

    Classification uses hierarchical macro recall: recall is calculated for
    each subject/class combination that exists, averaged across subjects within
    each class, and finally averaged across classes. A subject therefore never
    needs to contain every configured class. Regression remains the equally
    subject-weighted mean absolute error.
    """
    grouped = list(predictions.groupby("subject_id", sort=False))
    if not grouped:
        raise ExperimentError("Primary-score calculation received no predictions")

    if not task.is_classification:
        subject_errors = [
            float(
                np.mean(
                    np.abs(
                        group["y_pred"].to_numpy(float)
                        - group["y_true"].to_numpy(float)
                    )
                )
            )
            for _, group in grouped
        ]
        return float(np.mean(subject_errors)), len(grouped), 0

    class_scores: list[float] = []
    for label in range(task.class_count):
        subject_recalls = []
        for _, group in grouped:
            true = group["y_true"].to_numpy()
            mask = true == label
            if mask.any():
                predicted = group["y_pred"].to_numpy()
                subject_recalls.append(float(np.mean(predicted[mask] == label)))
        if subject_recalls:
            class_scores.append(float(np.mean(subject_recalls)))
    if not class_scores:
        raise ExperimentError("Primary-score calculation found no observed classes")
    return float(np.mean(class_scores)), len(grouped), 0
