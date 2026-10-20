from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from ..tasks import TaskDefinition
from ..utils import stable_seed


def _paired_test(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["subject_id", "sample_id"]
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Paired tests require one prediction per subject/sample")
    paired = left[[*keys, "y_true", "y_pred"]].merge(
        right[[*keys, "y_true", "y_pred"]],
        on=keys,
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    empty = {
        "paired_subjects": 0,
        "paired_samples": 0,
        "pooled_metric_a": math.nan,
        "pooled_metric_b": math.nan,
        "observed_difference_a_minus_b": math.nan,
        "permutation_extreme_count": math.nan,
        "p_value": math.nan,
    }
    if not (paired["_merge"] == "both").all():
        return {"pairing_status": "incompatible_sample_sets", **empty}
    true_a = paired["y_true_a"].to_numpy()
    true_b = paired["y_true_b"].to_numpy()
    if not np.array_equal(true_a, true_b):
        return {"pairing_status": "incompatible_true_labels", **empty}
    predicted_a = paired["y_pred_a"].to_numpy()
    predicted_b = paired["y_pred_b"].to_numpy()
    if metric == "subject_macro_mae":
        row_a = np.abs(predicted_a - true_a)
        row_b = np.abs(predicted_b - true_a)
        subject_errors = pd.DataFrame(
            {
                "subject_id": paired["subject_id"],
                "error_a": row_a,
                "error_b": row_b,
            }
        ).groupby("subject_id", sort=False).mean()
        pooled_a = float(subject_errors["error_a"].mean())
        pooled_b = float(subject_errors["error_b"].mean())
        contributions = (
            (subject_errors["error_a"] - subject_errors["error_b"])
            / len(subject_errors)
        ).to_numpy(float)
    elif metric == "subject_class_macro_recall":
        labels = np.unique(true_a)
        subject_contributions: dict[str, float] = {
            str(subject): 0.0 for subject in paired["subject_id"].unique()
        }
        pooled_a = 0.0
        pooled_b = 0.0
        for label in labels:
            label_rows = paired.loc[true_a == label].copy()
            recalls = label_rows.assign(
                correct_a=(predicted_a[true_a == label] == label).astype(float),
                correct_b=(predicted_b[true_a == label] == label).astype(float),
            ).groupby("subject_id", sort=False)[["correct_a", "correct_b"]].mean()
            pooled_a += float(recalls["correct_a"].mean()) / len(labels)
            pooled_b += float(recalls["correct_b"].mean()) / len(labels)
            weight = 1.0 / (len(labels) * len(recalls))
            for subject, values in recalls.iterrows():
                subject_contributions[str(subject)] += (
                    float(values["correct_a"] - values["correct_b"]) * weight
                )
        contributions = np.asarray(
            list(subject_contributions.values()), dtype=float
        )
    else:
        correct_a = (predicted_a == true_a).astype(float)
        correct_b = (predicted_b == true_a).astype(float)
        if metric == "accuracy":
            pooled_a, pooled_b = float(correct_a.mean()), float(correct_b.mean())
            row_contribution = (correct_a - correct_b) / len(paired)
        else:
            labels, inverse, counts = np.unique(true_a, return_inverse=True, return_counts=True)
            if len(labels) < 2:
                return {"pairing_status": "pooled_target_is_single_class", **empty}
            weights = 1.0 / (len(labels) * counts[inverse])
            pooled_a = float(np.sum(correct_a * weights))
            pooled_b = float(np.sum(correct_b * weights))
            row_contribution = (correct_a - correct_b) * weights
        contributions = (
            pd.DataFrame(
                {"subject_id": paired["subject_id"], "value": row_contribution}
            )
            .groupby("subject_id", sort=False)["value"]
            .sum()
            .to_numpy(float)
        )
    observed = float(contributions.sum())
    random = np.random.default_rng(seed)
    signs = random.integers(0, 2, size=(repeats, len(contributions)), dtype=np.int8) * 2 - 1
    permuted = signs @ contributions
    tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 8
    extreme = int(np.count_nonzero(np.abs(permuted) + tolerance >= abs(observed)))
    return {
        "pairing_status": "ok",
        "paired_subjects": len(contributions),
        "paired_samples": len(paired),
        "pooled_metric_a": pooled_a,
        "pooled_metric_b": pooled_b,
        "observed_difference_a_minus_b": observed,
        "permutation_extreme_count": extreme,
        "p_value": (extreme + 1) / (repeats + 1),
    }


def _holm(values: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.dropna().sort_values()
    running = 0.0
    count = len(finite)
    for rank, (index, value) in enumerate(finite.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        output.loc[index] = running
    return output


def _comparison_groups(
    predictions: pd.DataFrame,
) -> list[tuple[pd.DataFrame, str, str]]:
    groups: list[tuple[pd.DataFrame, str, str]] = []
    for keys, group in predictions.groupby(
        ["input_id", "target", "input_mode", "representation"], sort=False
    ):
        if group["model"].nunique() > 1:
            groups.append(
                (group, "models_within_representation", "model|" + "|".join(map(str, keys)))
            )
    for keys, group in predictions.groupby(
        ["input_id", "target", "input_mode", "model"], sort=False
    ):
        if group["representation"].nunique() > 1:
            groups.append(
                (
                    group,
                    "representations_within_model",
                    "representation|" + "|".join(map(str, keys)),
                )
            )
    for keys, group in predictions.groupby(
        ["target", "input_mode", "representation", "model"], sort=False
    ):
        if group["input_id"].nunique() > 1:
            groups.append(
                (group, "input_sensitivity", "input|" + "|".join(map(str, keys)))
            )
    for keys, group in predictions.groupby(
        ["input_id", "target", "representation", "model"], sort=False
    ):
        if group["input_mode"].nunique() > 1:
            groups.append(
                (
                    group,
                    "input_mode_sensitivity",
                    "input_mode|" + "|".join(map(str, keys)),
                )
            )
    return groups


def paired_statistical_test_count(
    predictions: pd.DataFrame, task: TaskDefinition
) -> int:
    """Return the exact number of candidate-pair/metric tests to run."""
    if predictions.empty:
        return 0
    metric_count = 2 if task.is_classification else 1
    return sum(
        math.comb(int(group["experiment_id"].nunique()), 2) * metric_count
        for group, _, _ in _comparison_groups(predictions)
    )


def paired_statistical_tests(
    predictions: pd.DataFrame,
    task: TaskDefinition,
    repeats: int,
    seed: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    metric_names = (
        ("subject_class_macro_recall", "accuracy")
        if task.is_classification
        else ("subject_macro_mae",)
    )
    rows: list[dict[str, Any]] = []
    total = paired_statistical_test_count(predictions, task)
    completed = 0

    def compare(group: pd.DataFrame, comparison_type: str, family: str) -> None:
        nonlocal completed
        experiment_ids = group["experiment_id"].unique()
        for left_id, right_id in itertools.combinations(experiment_ids, 2):
            left = group.loc[group["experiment_id"] == left_id]
            right = group.loc[group["experiment_id"] == right_id]
            for metric in metric_names:
                result = _paired_test(left, right, metric, repeats, stable_seed(seed, family, left_id, right_id, metric))
                rows.append(
                    {
                        "comparison_family": family,
                        "comparison_type": comparison_type,
                        "metric": metric,
                        "experiment_a": left_id,
                        "experiment_b": right_id,
                        "test_method": "paired_subject_cluster_permutation",
                        "input_a": left.iloc[0]["input_id"],
                        "input_b": right.iloc[0]["input_id"],
                        "target": left.iloc[0]["target"],
                        "input_mode_a": left.iloc[0]["input_mode"],
                        "input_mode_b": right.iloc[0]["input_mode"],
                        "representation_a": left.iloc[0]["representation"],
                        "representation_b": right.iloc[0]["representation"],
                        "model_a": left.iloc[0]["model"],
                        "model_b": right.iloc[0]["model"],
                        "permutation_repeats": repeats,
                        **result,
                    }
                )
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        total,
                        f"{comparison_type} | {metric} | {left_id} vs {right_id}",
                    )

    for group, comparison_type, family in _comparison_groups(predictions):
        compare(group, comparison_type, family)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_value_holm"] = result.groupby(["comparison_family", "metric"], group_keys=False)["p_value"].apply(_holm)
    result["significant_holm_0_05"] = result["p_value_holm"] < 0.05
    return result
