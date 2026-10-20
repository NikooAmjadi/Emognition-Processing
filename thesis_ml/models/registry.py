from __future__ import annotations

from typing import Callable

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config_types import AppConfig
from ..errors import ExperimentError
from ..tasks import TaskDefinition
from .base import ModelBundle, prefixed_grid
from .deep import TorchDenseEstimator
from .graph import GraphEstimator, graph_node_groups
from .sklearn_models import build_sklearn_model

SKLEARN_MODELS = {"svm", "rf", "knn", "xgboost", "mlp"}
DEEP_MODELS = {"torch_mlp"}
GRAPH_MODELS = {"gcn", "gat"}


def available_models() -> tuple[str, ...]:
    return tuple(sorted(SKLEARN_MODELS | DEEP_MODELS | GRAPH_MODELS))


def _preprocessing_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def build_model(
    name: str,
    task: TaskDefinition,
    features: list[str],
    representation: str,
    config: AppConfig,
    seed: int,
) -> ModelBundle:
    if name in SKLEARN_MODELS:
        return build_sklearn_model(name, task, config, seed)
    if name == "torch_mlp":
        estimator = TorchDenseEstimator(
            task_kind=task.config.kind,
            class_count=task.class_count,
            epochs=config.models.graph_max_epochs,
            batch_size=config.models.graph_batch_size,
            device=config.models.device,
            random_state=seed,
        )
        grid = (
            {"hidden_channels": [128], "dropout": [0.2], "learning_rate": [0.001], "weight_decay": [0.0]}
            if config.evaluation.quick
            else {"hidden_channels": [64, 128], "dropout": [0.2, 0.5], "learning_rate": [0.001, 0.0003], "weight_decay": [0.0, 0.0005]}
        )
        override = config.models.parameter_overrides.get(name)
        if override:
            grid = {key: list(values) for key, values in override.items()}
        return ModelBundle(_preprocessing_pipeline(estimator), prefixed_grid(grid), supports_shap=True)
    if name in GRAPH_MODELS:
        if representation not in config.models.graph_representations:
            raise ExperimentError(f"{name} does not support representation {representation}")
        groups = graph_node_groups(
            features,
            representation,
            config.schema.graph_node_tokens,
            config.schema.graph_node_aliases,
        )
        estimator = GraphEstimator(
            model_name=name,
            node_groups=groups,
            task_kind=task.config.kind,
            class_count=task.class_count,
            epochs=config.models.graph_max_epochs,
            min_epochs=config.models.graph_min_epochs,
            patience=config.models.graph_patience,
            min_delta=config.models.graph_min_delta,
            batch_size=config.models.graph_batch_size,
            eval_batch_size=config.models.graph_eval_batch_size,
            device=config.models.graph_device,
            random_state=seed,
        )
        grid = {
            "hidden_channels": [32],
            "dropout": [0.2],
            "learning_rate": [0.001] if config.evaluation.quick else [0.001, 0.0003],
            "weight_decay": [0.0],
        }
        if name == "gat":
            grid["heads"] = [2]
        override = config.models.parameter_overrides.get(name)
        if override:
            grid = {key: list(values) for key, values in override.items()}
        return ModelBundle(
            _preprocessing_pipeline(estimator),
            prefixed_grid(grid),
            supports_shap=False,
            is_graph=True,
        )
    raise ExperimentError(f"Unknown model {name!r}; available: {available_models()}")
