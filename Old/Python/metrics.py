from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def compute_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    classification: str,
) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    if classification == "regression":
        return {
            "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
            "rmse": float(
                np.sqrt(mean_squared_error(y_true_array, y_pred_array))
            ),
            "r2": float(r2_score(y_true_array, y_pred_array)),
        }

    return {
        "acc": float(accuracy_score(y_true_array, y_pred_array)),
        "f1": float(
            f1_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                y_pred_array,
                average="macro",
                zero_division=0,
            )
        ),
    }


def selection_score(
    metrics: dict[str, float],
    classification: str,
) -> float:
    if classification == "regression":
        return -metrics["mae"]
    return metrics["f1"]


def build_summary(
    model_fold_metrics: dict[str, list[dict[str, float]]],
    classification: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for model_name, fold_metrics in model_fold_metrics.items():
        if not fold_metrics:
            raise ValueError(
                f"No fold metrics were produced for model {model_name!r}."
            )

        if classification == "regression":
            row: dict[str, float | str] = {
                "Model": model_name,
                "MAE_mean": np.mean([m["mae"] for m in fold_metrics]),
                "MAE_std": np.std([m["mae"] for m in fold_metrics]),
                "RMSE_mean": np.mean([m["rmse"] for m in fold_metrics]),
                "RMSE_std": np.std([m["rmse"] for m in fold_metrics]),
                "R2_mean": np.mean([m["r2"] for m in fold_metrics]),
                "R2_std": np.std([m["r2"] for m in fold_metrics]),
            }
        else:
            row = {
                "Model": model_name,
                "Acc_mean": np.mean([m["acc"] for m in fold_metrics]),
                "Acc_std": np.std([m["acc"] for m in fold_metrics]),
                "F1_mean": np.mean([m["f1"] for m in fold_metrics]),
                "F1_std": np.std([m["f1"] for m in fold_metrics]),
                "Prec_mean": np.mean(
                    [m["precision"] for m in fold_metrics]
                ),
                "Prec_std": np.std(
                    [m["precision"] for m in fold_metrics]
                ),
                "Rec_mean": np.mean([m["recall"] for m in fold_metrics]),
                "Rec_std": np.std([m["recall"] for m in fold_metrics]),
            }

        rows.append(row)

    summary = pd.DataFrame(rows)
    if classification == "regression":
        return summary.sort_values("MAE_mean", ascending=True).reset_index(
            drop=True
        )

    return summary.sort_values("F1_mean", ascending=False).reset_index(
        drop=True
    )
