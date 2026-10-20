from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Dataset:
    dataset_id: str
    input_id: str
    path: Path
    sha256: str
    table: pd.DataFrame
    feature_columns: dict[str, list[str]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    order: int
    dataset: Dataset
    input_mode: str
    representation: str
    model_name: str


@dataclass
class RunResults:
    summaries: list[dict[str, Any]] = field(default_factory=list)
    folds: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    permutation_importance: list[dict[str, Any]] = field(default_factory=list)
    shap_importance: list[dict[str, Any]] = field(default_factory=list)
    shap_values: list[dict[str, Any]] = field(default_factory=list)
    inner_candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    inner_split_assignments: list[dict[str, Any]] = field(default_factory=list)
    tuned_candidates: list[dict[str, Any]] = field(default_factory=list)
    window_predictions: list[dict[str, Any]] = field(default_factory=list)

    def add_nested(self, result: "NestedTargetResult") -> None:
        self.summaries.extend(result.summaries)
        self.folds.extend(result.folds)
        self.predictions.extend(result.predictions)
        self.permutation_importance.extend(result.permutation_importance)
        self.shap_importance.extend(result.shap_importance)
        self.shap_values.extend(result.shap_values)
        self.inner_candidate_scores.extend(result.inner_candidate_scores)
        self.inner_split_assignments.extend(result.inner_split_assignments)
        self.tuned_candidates.extend(result.tuned_candidates)
        self.window_predictions.extend(result.window_predictions)


@dataclass
class NestedTargetResult:
    summaries: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    inner_candidate_scores: list[dict[str, Any]]
    inner_split_assignments: list[dict[str, Any]]
    tuned_candidates: list[dict[str, Any]]
    window_predictions: list[dict[str, Any]] = field(default_factory=list)
    permutation_importance: list[dict[str, Any]] = field(default_factory=list)
    shap_importance: list[dict[str, Any]] = field(default_factory=list)
    shap_values: list[dict[str, Any]] = field(default_factory=list)
