from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class GraphFeaturePreprocessor:
    """Fold-local imputation and scaling for graph node features."""

    def __init__(self) -> None:
        self._imputer = SimpleImputer(
            strategy="mean",
            keep_empty_features=True,
        )
        self._scaler = StandardScaler()
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> "GraphFeaturePreprocessor":
        if X.empty:
            raise ValueError("Cannot fit graph preprocessing on empty data.")

        self._columns = X.columns.tolist()
        imputed = self._imputer.fit_transform(X)
        self._scaler.fit(imputed)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._columns is None:
            raise RuntimeError("GraphFeaturePreprocessor must be fitted first.")

        missing = [column for column in self._columns if column not in X]
        extra = [column for column in X if column not in self._columns]
        if missing or extra:
            raise ValueError(
                "Graph feature columns changed between fit and transform. "
                f"Missing: {missing}; extra: {extra}"
            )

        ordered = X[self._columns]
        imputed = self._imputer.transform(ordered)
        scaled = self._scaler.transform(imputed)
        return pd.DataFrame(
            scaled,
            columns=self._columns,
            index=X.index,
        )

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
