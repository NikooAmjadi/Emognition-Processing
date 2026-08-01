from __future__ import annotations

import json
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, ParameterGrid

from config import Config
from graph_data.dataset import create_graph_loader
from graph_data.graph_builder import GraphBuilder
from graph_data.preprocessor import GraphFeaturePreprocessor
from metrics import compute_metrics, selection_score
from models.specs import ModelSpec
from runners.graph_trainer import (
    predict_graph_model,
    resolve_device,
    set_seed,
    train_graph_model,
)
from runners.types import ModelRunResult


def _threshold_used(config: Config) -> float | None:
    if config.classification != "binary":
        return None
    return 0.0 if config.normalize_labels else 5.0


def _make_loaders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    config: Config,
    feature_mode: str,
    input_mode: str,
):
    preprocessor = GraphFeaturePreprocessor()
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_validation = preprocessor.transform(X_validation)

    builder = GraphBuilder(config, feature_mode, input_mode)
    train_graphs = builder.build(transformed_train)
    validation_graphs = builder.build(transformed_validation)

    train_loader = create_graph_loader(
        train_graphs,
        y_train.to_numpy(),
        config.classification,
        config.graph_batch_size,
        True,
        config.graph_num_workers,
    )
    validation_loader = create_graph_loader(
        validation_graphs,
        y_validation.to_numpy(),
        config.classification,
        config.graph_eval_batch_size,
        False,
        config.graph_num_workers,
    )
    return train_loader, validation_loader, builder.schema


def _evaluate_parameter_set(
    spec: ModelSpec,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    config: Config,
    feature_mode: str,
    input_mode: str,
    fold_seed: int,
) -> tuple[float, int]:
    unique_groups = groups_train.nunique()
    inner_splits = min(config.inner_cv_splits, unique_groups)
    if inner_splits < 2:
        raise ValueError(
            "Not enough training subjects for graph inner GroupKFold."
        )

    scores: list[float] = []
    best_epochs: list[int] = []
    splitter = GroupKFold(n_splits=inner_splits)
    device = resolve_device(config)

    for inner_fold, (inner_train_idx, validation_idx) in enumerate(
        splitter.split(X_train, y_train, groups_train),
        start=1,
    ):
        set_seed(fold_seed + inner_fold)

        train_loader, validation_loader, schema = _make_loaders(
            X_train.iloc[inner_train_idx],
            y_train.iloc[inner_train_idx],
            X_train.iloc[validation_idx],
            y_train.iloc[validation_idx],
            config,
            feature_mode,
            input_mode,
        )
        if schema is None:
            raise RuntimeError("Graph schema was not created.")

        model = spec.factory(
            schema.features_per_node,
            config.output_channels,
            params,
        )
        training_result = train_graph_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            classification=config.classification,
            learning_rate=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
            config=config,
            device=device,
        )

        y_true, y_pred = predict_graph_model(
            model,
            validation_loader,
            config.classification,
            device,
        )
        metrics = compute_metrics(
            y_true,
            y_pred,
            config.classification,
        )
        scores.append(selection_score(metrics, config.classification))
        best_epochs.append(training_result.best_epoch)

    return float(np.mean(scores)), max(1, int(round(median(best_epochs))))


def run_graph_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    config: Config,
    feature_mode: str,
    input_mode: str,
) -> ModelRunResult:
    # Raises a clear configuration error before expensive CV starts.
    config.graph_prefixes_for(feature_mode)

    param_grid = spec.param_grid_factory(config)
    combinations = list(ParameterGrid(param_grid))
    print(f"\nRunning {spec.name} | Mode: {input_mode}")
    print(f"  Number of graph parameter combinations: {len(combinations)}")
    print(f"  Device: {resolve_device(config)}")

    logo = LeaveOneGroupOut()
    fold_metrics: list[dict[str, float]] = []
    subject_rows: list[dict[str, object]] = []

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
        groups_train = groups.iloc[train_index]

        best_params: dict[str, Any] | None = None
        best_score = -np.inf
        selected_epochs = config.graph_max_epochs

        for combination_index, params in enumerate(combinations, start=1):
            print(
                f"    Parameters {combination_index}/{len(combinations)}: "
                f"{params}"
            )
            score, epochs = _evaluate_parameter_set(
                spec=spec,
                params=params,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
                config=config,
                feature_mode=feature_mode,
                input_mode=input_mode,
                fold_seed=config.random_state + fold_index * 1000,
            )
            print(
                f"      Mean inner score={score:.6f} | "
                f"selected_epochs={epochs}"
            )

            if score > best_score:
                best_score = score
                best_params = dict(params)
                selected_epochs = epochs

        if best_params is None:
            raise RuntimeError(
                f"No valid parameter set was found for {spec.name}."
            )

        # Final preprocessing is fitted only on the complete outer training set.
        preprocessor = GraphFeaturePreprocessor()
        transformed_train = preprocessor.fit_transform(X_train)
        transformed_test = preprocessor.transform(X_test)

        builder = GraphBuilder(config, feature_mode, input_mode)
        train_graphs = builder.build(transformed_train)
        test_graphs = builder.build(transformed_test)
        if builder.schema is None:
            raise RuntimeError("Graph schema was not created.")

        train_loader = create_graph_loader(
            train_graphs,
            y_train.to_numpy(),
            config.classification,
            config.graph_batch_size,
            True,
            config.graph_num_workers,
        )
        test_loader = create_graph_loader(
            test_graphs,
            y_test.to_numpy(),
            config.classification,
            config.graph_eval_batch_size,
            False,
            config.graph_num_workers,
        )

        set_seed(config.random_state + fold_index)
        model = spec.factory(
            builder.schema.features_per_node,
            config.output_channels,
            best_params,
        )
        device = resolve_device(config)
        train_graph_model(
            model=model,
            train_loader=train_loader,
            validation_loader=None,
            classification=config.classification,
            learning_rate=float(best_params["learning_rate"]),
            weight_decay=float(best_params["weight_decay"]),
            config=config,
            device=device,
            fixed_epochs=selected_epochs,
        )

        y_true, y_pred = predict_graph_model(
            model,
            test_loader,
            config.classification,
            device,
        )
        metrics = compute_metrics(
            y_true,
            y_pred,
            config.classification,
        )
        fold_metrics.append(metrics)

        recorded_params = {
            **best_params,
            "selected_epochs": selected_epochs,
            "inner_score": best_score,
        }
        subject_rows.append(
            {
                "model": spec.name,
                "subject_id": subject_id,
                "threshold_used": _threshold_used(config),
                "neutral_label_removed": config.remove_baseline_neutral,
                "best_params": json.dumps(
                    recorded_params,
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
