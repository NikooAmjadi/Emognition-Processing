from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from model_registry import register_model
from models.specs import ModelSpec


def build_knn(config: Config) -> Pipeline:
    estimator = (
        KNeighborsRegressor()
        if config.classification == "regression"
        else KNeighborsClassifier()
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )


def knn_param_grid(config: Config) -> dict[str, list[object]]:
    if config.fast_mode:
        return {
            "clf__n_neighbors": [5],
            "clf__weights": ["distance"],
            "clf__p": [2],
        }
    return {
        "clf__n_neighbors": [3, 5, 7],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }


register_model(
    ModelSpec(
        name="KNN",
        family="tabular",
        factory=build_knn,
        param_grid_factory=knn_param_grid,
    )
)
