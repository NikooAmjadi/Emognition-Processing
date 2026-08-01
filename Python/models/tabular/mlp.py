from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


def build_mlp(config: Config) -> Pipeline:
    common = {
        "random_state": config.random_state,
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
    }

    estimator = (
        MLPRegressor(**common)
        if config.classification == "regression"
        else MLPClassifier(**common)
    )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def mlp_param_grid(config: Config) -> dict[str, list[object]]:
    if config.fast_mode:
        return {
            "clf__hidden_layer_sizes": [(128,)],
            "clf__alpha": [0.0001],
            "clf__learning_rate_init": [0.001],
        }
    return {
        "clf__hidden_layer_sizes": [
            (128,),
            (256, 128),
            (256, 128, 64),
        ],
        "clf__alpha": [0.0001, 0.001],
        "clf__learning_rate_init": [0.001, 0.0005],
    }


register_model(
    ModelSpec(
        name="MLP",
        family="tabular",
        factory=build_mlp,
        param_grid_factory=mlp_param_grid,
    )
)
