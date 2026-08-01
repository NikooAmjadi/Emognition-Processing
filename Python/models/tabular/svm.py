from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, SVR

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


def build_svm(config: Config) -> Pipeline:
    estimator = (
        SVR()
        if config.classification == "regression"
        else SVC(random_state=config.random_state)
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def svm_param_grid(config: Config) -> dict[str, list[object]]:
    if config.fast_mode:
        return {
            "clf__kernel": ["rbf"],
            "clf__C": [5],
            "clf__gamma": ["scale"],
        }
    return {
        "clf__kernel": ["rbf"],
        "clf__C": [1, 5, 10],
        "clf__gamma": ["scale"],
    }


register_model(
    ModelSpec(
        name="SVM",
        family="tabular",
        factory=build_svm,
        param_grid_factory=svm_param_grid,
    )
)
