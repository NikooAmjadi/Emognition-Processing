from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


def build_random_forest(config: Config) -> Pipeline:
    if config.classification == "regression":
        estimator = RandomForestRegressor(
            random_state=config.random_state,
            n_jobs=-1,
        )
    else:
        estimator = RandomForestClassifier(
            random_state=config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", estimator),
        ]
    )


def random_forest_param_grid(
    config: Config,
) -> dict[str, list[object]]:
    if config.fast_mode:
        return {
            "clf__n_estimators": [100],
            "clf__max_depth": [10],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["sqrt"],
        }
    return {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt"],
    }


register_model(
    ModelSpec(
        name="Random Forest",
        family="tabular",
        factory=build_random_forest,
        param_grid_factory=random_forest_param_grid,
    )
)
