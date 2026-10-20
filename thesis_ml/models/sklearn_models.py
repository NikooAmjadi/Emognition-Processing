from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight

from ..config_types import AppConfig
from ..errors import ExperimentError
from ..tasks import TaskDefinition
from .base import ModelBundle, prefixed_grid


class BalancedFitClassifier(ClassifierMixin, BaseEstimator):
    """Cloneable wrapper that applies balanced sample weights on every fit."""

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "BalancedFitClassifier":
        self.estimator_ = clone(self.estimator)
        labels = np.asarray(y)
        weights = compute_sample_weight(class_weight="balanced", y=labels)
        self.estimator_.fit(X, y, sample_weight=weights, **kwargs)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def decision_function(self, X: Any) -> np.ndarray:
        return self.estimator_.decision_function(X)


def _pipeline(classifier: Any, scale: bool = True) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(0.0)),
            ("scaler", StandardScaler() if scale else "passthrough"),
            ("clf", classifier),
        ]
    )


def build_sklearn_model(
    name: str, task: TaskDefinition, config: AppConfig, seed: int
) -> ModelBundle:
    classification = task.is_classification
    jobs = config.evaluation.n_jobs
    quick = config.evaluation.quick
    if name == "svm":
        estimator = (
            SVC(class_weight="balanced", random_state=seed)
            if classification
            else SVR()
        )
        grid = {"C": [1.0], "gamma": ["scale"]} if quick else {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}
        scale = True
    elif name == "rf":
        estimator = (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=jobs,
            )
            if classification
            else RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=jobs)
        )
        grid = (
            {"n_estimators": [300], "max_depth": [None], "min_samples_leaf": [1], "max_features": ["sqrt"]}
            if quick
            else {"n_estimators": [300], "max_depth": [None, 12], "min_samples_leaf": [1, 3], "max_features": ["sqrt"]}
        )
        scale = False
    elif name == "knn":
        estimator = KNeighborsClassifier(n_jobs=jobs) if classification else KNeighborsRegressor(n_jobs=jobs)
        grid = (
            {"n_neighbors": [5], "weights": ["distance"], "p": [2]}
            if quick
            else {"n_neighbors": [3, 5, 9], "weights": ["uniform", "distance"], "p": [2]}
        )
        scale = True
    elif name == "mlp":
        if classification:
            base = MLPClassifier(
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=seed,
            )
            estimator = BalancedFitClassifier(base)
            prefix = "estimator__"
        else:
            estimator = MLPRegressor(
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=seed,
            )
            prefix = ""
        grid = (
            {f"{prefix}hidden_layer_sizes": [(128,)], f"{prefix}alpha": [0.0001], f"{prefix}learning_rate_init": [0.001]}
            if quick
            else {
                f"{prefix}hidden_layer_sizes": [(128,), (256, 128), (256, 128, 64)],
                f"{prefix}alpha": [0.0001, 0.001],
                f"{prefix}learning_rate_init": [0.001, 0.0005],
            }
        )
        scale = True
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ModuleNotFoundError as exc:
            raise ExperimentError("xgboost is required for the xgboost model") from exc
        common = {
            "tree_method": "hist",
            "device": config.models.device,
            "random_state": seed,
            "n_jobs": jobs,
            "verbosity": 0,
        }
        if classification:
            objective = "binary:logistic" if task.class_count == 2 else "multi:softprob"
            base = XGBClassifier(objective=objective, eval_metric="logloss", **common)
            estimator = BalancedFitClassifier(base)
            prefix = "estimator__"
        else:
            estimator = XGBRegressor(objective="reg:squarederror", eval_metric="rmse", **common)
            prefix = ""
        grid = (
            {
                f"{prefix}n_estimators": [250],
                f"{prefix}max_depth": [3],
                f"{prefix}learning_rate": [0.05],
                f"{prefix}subsample": [0.85],
                f"{prefix}colsample_bytree": [0.85],
            }
            if quick
            else {
                f"{prefix}n_estimators": [250, 450],
                f"{prefix}max_depth": [3, 5],
                f"{prefix}learning_rate": [0.03, 0.08],
                f"{prefix}subsample": [0.85],
                f"{prefix}colsample_bytree": [0.85],
            }
        )
        scale = False
    else:
        raise ExperimentError(f"Unknown sklearn model: {name}")

    override = config.models.parameter_overrides.get(name)
    if override:
        grid = {key: list(values) for key, values in override.items()}
    return ModelBundle(_pipeline(estimator, scale), prefixed_grid(grid))
