from __future__ import annotations

from importlib import import_module

from config import Config
from models.specs import ModelSpec


_MODEL_REGISTRY: dict[str, ModelSpec] = {}

# Lazy imports are important: users who run only sklearn models must not need
# torch-geometric installed.
_MODEL_MODULES: dict[str, str] = {
    "SVM": "models.tabular.svm",
    "Random Forest": "models.tabular.random_forest",
    "KNN": "models.tabular.knn",
    "XGBoost": "models.tabular.xgboost_model",
    "MLP": "models.tabular.mlp",
    "GCN": "models.graph.gcn",
    "GAT": "models.graph.gat",
}


def register_model(spec: ModelSpec) -> ModelSpec:
    if spec.name in _MODEL_REGISTRY:
        raise ValueError(f"Model {spec.name!r} is already registered.")
    _MODEL_REGISTRY[spec.name] = spec
    return spec


def _ensure_registered(name: str) -> None:
    if name in _MODEL_REGISTRY:
        return

    module_name = _MODEL_MODULES.get(name)
    if module_name is None:
        raise ValueError(
            f"Unknown model {name!r}. Available models: "
            f"{sorted(_MODEL_MODULES)}"
        )

    try:
        import_module(module_name)
    except ModuleNotFoundError as exc:
        if name in {"GCN", "GAT"} and exc.name == "torch_geometric":
            raise ModuleNotFoundError(
                f"Model {name} requires torch-geometric. Install the graph "
                "dependencies listed in requirements-graph.txt."
            ) from exc
        raise

    if name not in _MODEL_REGISTRY:
        raise RuntimeError(
            f"Module {module_name!r} did not register model {name!r}."
        )


def get_model_specs(config: Config) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for name in config.models_to_run:
        _ensure_registered(name)
        specs.append(_MODEL_REGISTRY[name])
    return specs


def available_models() -> list[str]:
    return sorted(_MODEL_MODULES)
