from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


def build_xgboost(config: Config) -> Pipeline:
    common = {
        "random_state": config.random_state,
        "tree_method": "hist",
        "device": config.xgb_device,
        "n_jobs": config.n_jobs_xgb,
        "verbosity": 0,
    }

    if config.classification == "regression":
        estimator = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common,
        )
    elif config.classification == "binary":
        estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **common,
        )
    elif config.classification == "ternary":
        estimator = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **common,
        )
    else:
        raise ValueError(
            f"Unsupported classification: {config.classification!r}"
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", estimator),
        ]
    )


def xgboost_param_grid(config: Config) -> dict[str, list[object]]:
    if config.fast_mode:
        return {
            "clf__n_estimators": [100],
            "clf__learning_rate": [0.1],
            "clf__max_depth": [3],
            "clf__min_child_weight": [1],
            "clf__subsample": [0.85],
            "clf__colsample_bytree": [0.85],
            "clf__reg_alpha": [0],
            "clf__reg_lambda": [1],
        }
    return {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [2, 3],
        "clf__min_child_weight": [1, 3],
        "clf__subsample": [0.85],
        "clf__colsample_bytree": [0.85],
        "clf__reg_alpha": [0, 0.01],
        "clf__reg_lambda": [1, 2],
    }


register_model(
    ModelSpec(
        name="XGBoost",
        family="tabular",
        factory=build_xgboost,
        param_grid_factory=xgboost_param_grid,
    )
)
