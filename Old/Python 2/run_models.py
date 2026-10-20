#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.3.4",
#   "pandas==2.3.3",
#   "scikit-learn==1.9.0",
#   "scipy==1.17.1",
#   "shap==0.51.0",
#   "torch==2.8.0",
#   "torch-geometric==2.6.1",
#   "xgboost==3.2.0",
# ]
# ///

"""Train and evaluate models on extracted Emognition features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import shap
    import torch
    import torch.nn.functional as torch_f
    import torch_geometric
    import xgboost
    from torch import nn
    from torch.optim import Adam
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GraphDataLoader
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        GroupKFold,
        LeaveOneGroupOut,
        ParameterGrid,
        StratifiedGroupKFold,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.utils.validation import check_is_fitted
    from sklearn.metrics import make_scorer
except ModuleNotFoundError as exc:  # pragma: no cover - dependency bootstrap
    raise SystemExit(
        "Missing a required package. Use Python 3.11 and install exactly:\n"
        "  python -m pip install numpy==2.3.4 pandas==2.3.3 "
        "scikit-learn==1.9.0 scipy==1.17.1 shap==0.51.0 "
        "torch==2.8.0 torch-geometric==2.6.1 xgboost==3.2.0\n"
        f"Original import error: {exc}"
    ) from exc


GENERATOR_NAME = "run_models.py"
GENERATOR_VERSION = "1.5.0"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = SCRIPT_DIR / "features"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "model_results"
EXPECTED_FEATURE_COUNTS = {"hc": 29, "nvg": 48, "hvg": 48}
REQUIRED_METADATA = {
    "subject_id",
    "emotion",
    "movie_order_index",
    "window_id",
    "window_size_sec",
    "overlap_sec",
    "window_start_sec",
    "window_end_sec",
    "sam_valence",
    "sam_arousal",
}
FAMILIES = ("hc", "nvg", "hvg")
REPRESENTATIONS: dict[str, tuple[str, ...]] = {
    "hc": ("hc",),
    "nvg": ("nvg",),
    "hvg": ("hvg",),
    "hc+nvg": ("hc", "nvg"),
    "hc+hvg": ("hc", "hvg"),
}
TABULAR_MODEL_NAMES = ("svm", "rf", "knn", "xgboost", "mlp")
GRAPH_MODEL_NAMES = ("gcn", "gat")
MODEL_NAMES = TABULAR_MODEL_NAMES + GRAPH_MODEL_NAMES
GRAPH_REPRESENTATIONS = ("nvg", "hvg")
TARGETS = ("valence", "arousal")
INPUT_MODES = ("windowed", "aggregated")


def strict_binary_balanced_accuracy(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> float:
    """Return binary BA only when both target classes are observable."""
    true = np.asarray(y_true, dtype=int)
    if not np.array_equal(np.unique(true), np.array([0, 1])):
        return math.nan
    return float(
        recall_score(
            true,
            np.asarray(y_pred, dtype=int),
            average="macro",
            labels=[0, 1],
            zero_division=0,
        )
    )


STRICT_BALANCED_ACCURACY = make_scorer(strict_binary_balanced_accuracy)


class SimulationError(RuntimeError):
    pass


class AutoBalancedXGBClassifier(xgboost.XGBClassifier):
    """Update the XGBoost class weight for each fit."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "AutoBalancedXGBClassifier":
        labels = np.asarray(y, dtype=int)
        negatives = int((labels == 0).sum())
        positives = int((labels == 1).sum())
        if negatives == 0 or positives == 0:
            raise ValueError("XGBoost training data must contain both classes")
        self.set_params(scale_pos_weight=negatives / positives)
        return super().fit(X, y, **kwargs)


class AutoBalancedMLPClassifier(MLPClassifier):
    """Apply balanced sample weights independently for every fitted partition."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "AutoBalancedMLPClassifier":
        labels = np.asarray(y, dtype=int)
        if np.unique(labels).size != 2:
            raise ValueError("MLP training data must contain both classes")
        sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


@dataclass
class InputDataset:
    input_id: str
    path: Path
    sha256: str
    manifest_path: Path | None
    manifest_sha256: str | None
    table: pd.DataFrame
    feature_columns: dict[str, list[str]]
    window_sec: float
    overlap_sec: float


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: InputDataset
    target: str
    input_mode: str
    representation: str
    model_name: str


@dataclass
class ExperimentResults:
    summaries: list[dict[str, Any]]
    folds: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    importance: list[dict[str, Any]]
    shap_importance: list[dict[str, Any]]
    shap_values: list[dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible nested-LOSO Emognition binary classification."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help=(
            "Feature CSV from extract_emognition_features.py; repeat for "
            "window/overlap sensitivity. Default: all matching CSVs in ./features."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Parent output directory (default: ./model_results)."
    )
    parser.add_argument(
        "--run-name",
        help="Output subdirectory name. Default: UTC timestamp plus configuration hash."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--targets", nargs="+", choices=TARGETS, default=list(TARGETS)
    )
    parser.add_argument(
        "--input-modes", nargs="+", choices=INPUT_MODES, default=list(INPUT_MODES)
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=tuple(REPRESENTATIONS),
        default=list(REPRESENTATIONS),
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument(
        "--trial-policy",
        choices=("common", "available"),
        default="common",
        help=(
            "For multiple settings, use only trials shared by every input (common) "
            "or all trials available per setting (available)."
        ),
    )
    parser.add_argument(
        "--min-valid-trials-per-subject",
        type=int,
        default=6,
        help=(
            "Exclude a subject from every analysis when they have fewer than this "
            "many valid non-BASELINE/non-NEUTRAL trials in any input setting "
            "(default: 6; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--include-neutral-condition",
        action="store_true",
        help=(
            "Sensitivity option: retain the NEUTRAL film condition. Its SAM=5 rating "
            "is still excluded target-wise. BASELINE is always excluded."
        ),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument(
        "--statistical-permutation-repeats",
        type=int,
        default=10_000,
        help=(
            "Subject-clustered paired permutations used for model comparisons "
            "(default: 10000)."
        ),
    )
    parser.add_argument(
        "--importance-models",
        nargs="+",
        choices=TABULAR_MODEL_NAMES,
        default=["xgboost"],
        help="Models for held-out permutation importance (default: xgboost)."
    )
    parser.add_argument(
        "--shap-models",
        nargs="+",
        choices=TABULAR_MODEL_NAMES,
        default=["xgboost"],
        help=(
            "Models for held-out SHAP explanations (default: xgboost). "
            "TreeSHAP is used for RF/XGBoost and model-agnostic permutation "
            "SHAP for SVM/KNN/MLP. Graph models use neither path."
        ),
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Disable held-out SHAP computation entirely.",
    )
    parser.add_argument(
        "--shap-background-size",
        type=int,
        default=100,
        help="Maximum outer-training samples used as SHAP background (default: 100).",
    )
    parser.add_argument(
        "--shap-max-test-samples",
        type=int,
        default=0,
        help=(
            "Maximum held-out samples explained per subject; zero explains every "
            "test sample (default: 0)."
        ),
    )
    parser.add_argument(
        "--shap-top-k",
        type=int,
        default=10,
        help="Top-k cutoff used for cross-subject rank stability (default: 10).",
    )
    parser.add_argument(
        "--save-shap-values",
        action="store_true",
        help=(
            "Also save local sample-by-feature SHAP values. This can create a very "
            "large CSV; fold and cross-subject summaries are always saved."
        ),
    )
    parser.add_argument(
        "--no-correlations", action="store_true",
        help="Skip descriptive Spearman feature-correlation output."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use one deterministic parameter set per model; diagnostic only."
    )
    parser.add_argument(
        "--max-outer-folds",
        type=int,
        default=0,
        help="Limit LOSO folds for a smoke test. Zero means complete LOSO."
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu",
        help="XGBoost device. CPU is the reproducibility default."
    )
    parser.add_argument(
        "--graph-device", choices=("auto", "cpu", "cuda"), default="auto",
        help="PyTorch device for GCN/GAT (default: auto).",
    )
    parser.add_argument(
        "--graph-batch-size", type=int, default=32,
        help="Training batch size for GCN/GAT (default: 32).",
    )
    parser.add_argument(
        "--graph-eval-batch-size", type=int, default=128,
        help="Evaluation batch size for GCN/GAT (default: 128).",
    )
    parser.add_argument(
        "--graph-max-epochs", type=int, default=100,
        help="Maximum inner-CV epochs for GCN/GAT (default: 100).",
    )
    parser.add_argument(
        "--graph-min-epochs", type=int, default=5,
        help=(
            "Minimum epochs before graph early stopping can select a checkpoint "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--graph-patience", type=int, default=30,
        help="Validation-loss early-stopping patience for GCN/GAT (default: 30).",
    )
    parser.add_argument(
        "--graph-min-delta", type=float, default=1e-5,
        help="Minimum graph validation-loss improvement (default: 1e-5).",
    )
    return parser.parse_args(argv)


def unique_in_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def validate_args(args: argparse.Namespace) -> None:
    args.targets = unique_in_order(args.targets)
    args.input_modes = unique_in_order(args.input_modes)
    args.representations = unique_in_order(args.representations)
    args.models = unique_in_order(args.models)
    args.importance_models = unique_in_order(args.importance_models)
    args.shap_models = unique_in_order(args.shap_models)
    if args.inner_splits < 2:
        raise SimulationError("--inner-splits must be at least 2")
    if args.bootstrap_repeats < 0:
        raise SimulationError("--bootstrap-repeats cannot be negative")
    if args.permutation_repeats < 0:
        raise SimulationError("--permutation-repeats cannot be negative")
    if args.statistical_permutation_repeats < 1:
        raise SimulationError("--statistical-permutation-repeats must be at least 1")
    if args.shap_background_size < 1:
        raise SimulationError("--shap-background-size must be at least 1")
    if args.shap_max_test_samples < 0:
        raise SimulationError("--shap-max-test-samples cannot be negative")
    if args.shap_top_k < 1:
        raise SimulationError("--shap-top-k must be at least 1")
    if args.max_outer_folds < 0:
        raise SimulationError("--max-outer-folds cannot be negative")
    if args.min_valid_trials_per_subject < 0:
        raise SimulationError("--min-valid-trials-per-subject cannot be negative")
    if args.n_jobs == 0:
        raise SimulationError("--n-jobs cannot be zero")
    if args.graph_batch_size < 1:
        raise SimulationError("--graph-batch-size must be at least 1")
    if args.graph_eval_batch_size < 1:
        raise SimulationError("--graph-eval-batch-size must be at least 1")
    if args.graph_max_epochs < 1:
        raise SimulationError("--graph-max-epochs must be at least 1")
    if args.graph_min_epochs < 1:
        raise SimulationError("--graph-min-epochs must be at least 1")
    if args.graph_min_epochs > args.graph_max_epochs:
        raise SimulationError(
            "--graph-min-epochs cannot exceed --graph-max-epochs"
        )
    if args.graph_patience < 1:
        raise SimulationError("--graph-patience must be at least 1")
    if args.graph_min_delta < 0:
        raise SimulationError("--graph-min-delta cannot be negative")
    if set(args.models) & set(GRAPH_MODEL_NAMES) and not (
        set(args.representations) & set(GRAPH_REPRESENTATIONS)
    ):
        raise SimulationError(
            "GCN/GAT require at least one of these representations: "
            + ", ".join(GRAPH_REPRESENTATIONS)
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    value = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")
    return value % (2**31 - 1)


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("._")
    return token or "input"


def atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".building")
    try:
        table.to_csv(temporary, index=False, float_format="%.12g", na_rep="NaN")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".building")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    candidates = args.input or sorted(
        DEFAULT_FEATURE_DIR.glob("emognition_features_w*_o*.csv")
    )
    if not candidates:
        raise SimulationError(
            "No feature CSV was found. Wait for extract_emognition_features.py to "
            f"finish, or pass --input explicitly. Looked in: {DEFAULT_FEATURE_DIR}"
        )
    resolved: list[Path] = []
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise SimulationError(f"Input feature CSV does not exist: {path}")
        if path in resolved:
            raise SimulationError(f"Input feature CSV was supplied more than once: {path}")
        resolved.append(path)
    return resolved


def trial_id_series(table: pd.DataFrame) -> pd.Series:
    return (
        table["subject_id"].astype(str)
        + "|"
        + table["movie_order_index"].astype(str)
        + "|"
        + table["emotion"].astype(str).str.upper()
    )


def validate_manifest(manifest_path: Path, path: Path, table: pd.DataFrame, feature_columns: dict[str, list[str]]) -> str:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimulationError(f"Cannot read valid manifest {manifest_path}: {exc}") from exc
    output = manifest.get("output", {})
    expected_counts = {
        "handcrafted": EXPECTED_FEATURE_COUNTS["hc"],
        "nvg": EXPECTED_FEATURE_COUNTS["nvg"],
        "hvg": EXPECTED_FEATURE_COUNTS["hvg"],
    }
    if output.get("csv") != path.name:
        raise SimulationError(
            f"Manifest {manifest_path.name} describes {output.get('csv')!r}, not {path.name!r}"
        )
    if int(output.get("row_count", -1)) != len(table):
        raise SimulationError(
            f"Manifest/CSV row-count mismatch for {path.name}: "
            f"{output.get('row_count')} versus {len(table)}"
        )
    if int(output.get("column_count", -1)) != len(table.columns):
        raise SimulationError(
            f"Manifest/CSV column-count mismatch for {path.name}: "
            f"{output.get('column_count')} versus {len(table.columns)}"
        )
    if output.get("feature_counts_by_family") != expected_counts:
        raise SimulationError(
            f"Unexpected feature schema in {manifest_path.name}; expected "
            f"{expected_counts}, got {output.get('feature_counts_by_family')}"
        )
    actual_features = sorted(sum(feature_columns.values(), []))
    manifest_features = sorted(output.get("feature_columns", []))
    if actual_features != manifest_features:
        raise SimulationError(f"Manifest/CSV feature-column mismatch for {path.name}")
    return sha256_file(manifest_path)


def load_input(path: Path, include_neutral: bool) -> InputDataset:
    print(f"Reading and validating: {path}", flush=True)
    table = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_METADATA - set(table.columns)
    if missing:
        raise SimulationError(
            f"{path.name} is missing required columns: {', '.join(sorted(missing))}"
        )
    if table.empty:
        raise SimulationError(f"{path.name} contains no rows")
    if table.columns.duplicated().any():
        duplicates = table.columns[table.columns.duplicated()].tolist()
        raise SimulationError(f"{path.name} has duplicate columns: {duplicates}")

    feature_columns = {
        family: sorted(column for column in table.columns if column.startswith(family + "_"))
        for family in FAMILIES
    }
    counts = {family: len(columns) for family, columns in feature_columns.items()}
    if counts != EXPECTED_FEATURE_COUNTS:
        raise SimulationError(
            f"{path.name} has feature counts {counts}, but the required schema is "
            f"{EXPECTED_FEATURE_COUNTS}. Re-run the current "
            "extract_emognition_features.py; do not use a legacy feature CSV."
        )
    all_features = sum(feature_columns.values(), [])
    numeric_features = table[all_features].apply(pd.to_numeric, errors="coerce")
    coercion_failures = int(
        ((numeric_features.isna()) & (~table[all_features].isna())).to_numpy().sum()
    )
    if coercion_failures:
        raise SimulationError(
            f"{path.name} contains {coercion_failures} non-numeric feature values"
        )
    if np.isinf(numeric_features.to_numpy(dtype=float)).any():
        raise SimulationError(f"{path.name} contains infinite feature values")
    table.loc[:, all_features] = numeric_features

    for column in ("sam_valence", "sam_arousal"):
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or not values.between(1, 9).all() or not np.allclose(values, np.round(values)):
            raise SimulationError(f"{path.name}: {column} must contain integer SAM values 1--9")
        table[column] = values.astype(int)
    table["emotion"] = table["emotion"].astype(str).str.strip().str.upper()

    manifest_path = path.with_suffix(".manifest.json")
    manifest_sha = None
    if manifest_path.is_file():
        manifest_sha = validate_manifest(
            manifest_path, path, table, feature_columns
        )
    else:
        manifest_path = None
        print(f"  Warning: no sibling manifest found for {path.name}", flush=True)

    table["trial_id"] = trial_id_series(table)
    duplicate_key = ["subject_id", "movie_order_index", "emotion", "window_id"]
    if table.duplicated(duplicate_key).any():
        raise SimulationError(f"{path.name} has duplicate window keys: {duplicate_key}")
    label_nunique = table.groupby("trial_id")[["sam_valence", "sam_arousal"]].nunique()
    if (label_nunique > 1).any(axis=None):
        raise SimulationError(f"{path.name} has inconsistent SAM labels within at least one trial")

    window_values = pd.to_numeric(table["window_size_sec"], errors="coerce").dropna().unique()
    overlap_values = pd.to_numeric(table["overlap_sec"], errors="coerce").dropna().unique()
    if len(window_values) != 1 or len(overlap_values) != 1:
        raise SimulationError(
            f"{path.name} must describe exactly one window/overlap setting"
        )
    window_sec, overlap_sec = float(window_values[0]), float(overlap_values[0])
    if window_sec <= 0 or overlap_sec < 0 or overlap_sec >= window_sec:
        raise SimulationError(f"{path.name} contains an invalid window/overlap setting")

    excluded = {"BASELINE"}
    if not include_neutral:
        excluded.add("NEUTRAL")
    table = table.loc[~table["emotion"].isin(excluded)].copy()
    if table.empty:
        raise SimulationError(f"{path.name}: condition filtering removed every row")

    input_id = safe_token(path.stem)
    print(
        f"  OK: {len(table):,} eligible windows, {table['trial_id'].nunique():,} trials, "
        f"{table['subject_id'].nunique()} subjects; w={window_sec:g}, o={overlap_sec:g}",
        flush=True,
    )
    return InputDataset(
        input_id=input_id,
        path=path,
        sha256=sha256_file(path),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        table=table,
        feature_columns=feature_columns,
        window_sec=window_sec,
        overlap_sec=overlap_sec,
    )


def apply_common_trial_policy(inputs: list[InputDataset], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_sets = [set(item.table["trial_id"]) for item in inputs]
    common = set.intersection(*trial_sets)
    if len(inputs) > 1 and common:
        reference = (
            inputs[0]
            .table.drop_duplicates("trial_id")
            .set_index("trial_id")[["sam_valence", "sam_arousal"]]
        )
        for item in inputs[1:]:
            candidate = (
                item.table.drop_duplicates("trial_id")
                .set_index("trial_id")[["sam_valence", "sam_arousal"]]
            )
            shared = sorted(common)
            if not reference.loc[shared].equals(candidate.loc[shared]):
                raise SimulationError(
                    "SAM labels disagree between sensitivity inputs for at least one "
                    f"common trial ({inputs[0].input_id} versus {item.input_id})"
                )
    for item, available in zip(inputs, trial_sets):
        before_rows = len(item.table)
        before_trials = len(available)
        if policy == "common":
            item.table = item.table.loc[item.table["trial_id"].isin(common)].copy()
        rows.append(
            {
                "input_id": item.input_id,
                "window_sec": item.window_sec,
                "overlap_sec": item.overlap_sec,
                "trial_policy": policy,
                "windows_before_policy": before_rows,
                "trials_before_policy": before_trials,
                "windows_after_policy": len(item.table),
                "trials_after_policy": item.table["trial_id"].nunique(),
                "common_trials_across_inputs": len(common),
            }
        )
        if item.table.empty:
            raise SimulationError(
                f"No trials remain for {item.input_id} under --trial-policy {policy}"
            )
    return rows


def subject_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def apply_subject_completeness_policy(inputs: list[InputDataset], minimum_trials: int) -> list[dict[str, Any]]:
    counts_by_input: dict[str, dict[str, int]] = {}
    all_subjects: set[str] = set()
    for item in inputs:
        # Neutral trials do not count toward coverage.
        coverage = item.table.loc[item.table["emotion"] != "NEUTRAL"]
        counts = (
            coverage.drop_duplicates(["subject_id", "trial_id"])
            .assign(_subject=lambda frame: frame["subject_id"].astype(str))
            .groupby("_subject")["trial_id"]
            .nunique()
            .astype(int)
            .to_dict()
        )
        counts_by_input[item.input_id] = counts
        all_subjects.update(counts)

    rows: list[dict[str, Any]] = []
    eligible_subjects: set[str] = set()
    excluded_subjects: list[str] = []
    for subject in sorted(all_subjects, key=subject_sort_key):
        per_input = {
            item.input_id: int(counts_by_input[item.input_id].get(subject, 0))
            for item in inputs
        }
        minimum_observed = min(per_input.values())
        eligible = minimum_trials == 0 or minimum_observed >= minimum_trials
        if eligible:
            eligible_subjects.add(subject)
        else:
            excluded_subjects.append(subject)
        row: dict[str, Any] = {
            "subject_id": subject,
            "minimum_required_valid_trials": minimum_trials,
            "minimum_valid_trials_across_inputs": minimum_observed,
            "eligible": eligible,
            "exclusion_reason": (
                ""
                if eligible
                else f"fewer_than_{minimum_trials}_valid_nonbaseline_nonneutral_trials"
            ),
        }
        for input_id, count in per_input.items():
            row[f"valid_trials__{input_id}"] = count
        rows.append(row)

    if not eligible_subjects:
        raise SimulationError(
            "Subject-completeness filtering removed every participant; lower "
            "--min-valid-trials-per-subject"
        )
    for item in inputs:
        item.table = item.table.loc[
            item.table["subject_id"].astype(str).isin(eligible_subjects)
        ].copy()
    if excluded_subjects:
        details = ", ".join(
            f"{subject} (min={min(counts_by_input[item.input_id].get(subject, 0) for item in inputs)})"
            for subject in excluded_subjects
        )
        print(
            "Subject-completeness exclusion: " + details,
            flush=True,
        )
    else:
        print("Subject-completeness exclusion: none", flush=True)
    return rows


def distribution_rows(item: InputDataset, target: str) -> list[dict[str, Any]]:
    label_column = f"sam_{target}"
    output: list[dict[str, Any]] = []
    for level, frame in (
        ("window", item.table),
        ("trial", item.table.drop_duplicates("trial_id")),
    ):
        raw_counts = frame[label_column].value_counts().sort_index()
        for sam in range(1, 10):
            output.append(
                {
                    "input_id": item.input_id,
                    "window_sec": item.window_sec,
                    "overlap_sec": item.overlap_sec,
                    "target": target,
                    "level": level,
                    "stage": "raw_sam_after_condition_filter",
                    "label": f"SAM_{sam}",
                    "count": int(raw_counts.get(sam, 0)),
                }
            )
        eligible = frame.loc[frame[label_column] != 5]
        binary = (eligible[label_column] > 5).astype(int)
        for label, name in ((0, "low_1_4"), (1, "high_6_9")):
            output.append(
                {
                    "input_id": item.input_id,
                    "window_sec": item.window_sec,
                    "overlap_sec": item.overlap_sec,
                    "target": target,
                    "level": level,
                    "stage": "binary_after_target_specific_SAM5_exclusion",
                    "label": name,
                    "count": int((binary == label).sum()),
                }
            )
    return output


def descriptive_correlations(item: InputDataset) -> pd.DataFrame:
    features = sum(item.feature_columns.values(), [])
    corr = item.table[features].corr(method="spearman", min_periods=3)
    rows: list[dict[str, Any]] = []
    family_of = {feature: feature.split("_", 1)[0] for feature in features}
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            rho = corr.at[left, right]
            rows.append(
                {
                    "input_id": item.input_id,
                    "window_sec": item.window_sec,
                    "overlap_sec": item.overlap_sec,
                    "feature_a": left,
                    "family_a": family_of[left],
                    "feature_b": right,
                    "family_b": family_of[right],
                    "same_family": family_of[left] == family_of[right],
                    "spearman_rho": float(rho) if pd.notna(rho) else math.nan,
                    "abs_spearman_rho": abs(float(rho)) if pd.notna(rho) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def representation_columns(item: InputDataset, representation: str) -> list[str]:
    return sum((item.feature_columns[family] for family in REPRESENTATIONS[representation]), [])


def target_table(item: InputDataset, target: str, input_mode: str, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    label_column = f"sam_{target}"
    frame = item.table.loc[item.table[label_column] != 5].copy()
    frame["target_sam"] = frame[label_column].astype(int)
    frame["y"] = (frame["target_sam"] > 5).astype(int)
    if frame["y"].nunique() != 2:
        raise SimulationError(
            f"{item.input_id}/{target}: both binary classes are not present after SAM=5 exclusion"
        )
    if input_mode == "windowed":
        frame["sample_id"] = (
            frame["trial_id"] + "|w" + frame["window_id"].astype(str)
        )
        return frame, feature_columns

    group_columns = ["subject_id", "emotion", "movie_order_index", "trial_id"]
    consistency = frame.groupby(group_columns)[["target_sam", "y"]].nunique()
    if (consistency > 1).any(axis=None):
        raise SimulationError(f"{item.input_id}/{target}: inconsistent trial labels")
    grouped = frame.groupby(group_columns, sort=False, dropna=False)
    means = grouped[feature_columns].mean().add_suffix("__mean")
    stds = grouped[feature_columns].std(ddof=0).add_suffix("__std")
    labels = grouped[["target_sam", "y"]].first()
    aggregated = pd.concat([labels, means, stds], axis=1).reset_index()
    aggregated["window_id"] = -1
    aggregated["window_start_sec"] = math.nan
    aggregated["window_end_sec"] = math.nan
    aggregated["sample_id"] = aggregated["trial_id"]
    aggregated_features = [
        *(feature + "__mean" for feature in feature_columns),
        *(feature + "__std" for feature in feature_columns),
    ]
    return aggregated, aggregated_features


def model_spec(model_name: str, seed: int, n_jobs: int, device: str, quick: bool) -> tuple[Pipeline, dict[str, list[Any]]]:
    if model_name == "svm":
        classifier = SVC(class_weight="balanced", kernel="rbf", random_state=seed)
        grid = (
            {"clf__C": [1.0], "clf__gamma": ["scale"]}
            if quick
            else {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]}
        )
        scaler: Any = StandardScaler()
    elif model_name == "rf":
        classifier = RandomForestClassifier(
            class_weight="balanced_subsample",
            criterion="gini",
            random_state=seed,
            n_jobs=n_jobs,
        )
        grid = (
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [None],
                "clf__min_samples_leaf": [1],
                "clf__max_features": ["sqrt"],
            }
            if quick
            else {
                "clf__n_estimators": [300],
                "clf__max_depth": [None, 12],
                "clf__min_samples_leaf": [1, 3],
                "clf__max_features": ["sqrt"],
            }
        )
        scaler = "passthrough"
    elif model_name == "knn":
        classifier = KNeighborsClassifier(n_jobs=n_jobs)
        grid = (
            {"clf__n_neighbors": [5], "clf__weights": ["distance"], "clf__p": [2]}
            if quick
            else {
                "clf__n_neighbors": [3, 5, 9],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [2],
            }
        )
        scaler = StandardScaler()
    elif model_name == "mlp":
        classifier = AutoBalancedMLPClassifier(
            random_state=seed,
            max_iter=2_000,
            solver="adam",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            tol=1e-4,
        )
        grid = (
            {
                "clf__hidden_layer_sizes": [(128,)],
                "clf__alpha": [0.0001],
                "clf__learning_rate_init": [0.001],
            }
            if quick
            else {
                "clf__hidden_layer_sizes": [(128,), (256, 128), (256, 128, 64)],
                "clf__alpha": [0.0001, 0.001],
                "clf__learning_rate_init": [0.001, 0.0005],
            }
        )
        scaler = StandardScaler()
    elif model_name == "xgboost":
        classifier = AutoBalancedXGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=device,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=0,
        )
        grid = (
            {
                "clf__n_estimators": [250],
                "clf__max_depth": [3],
                "clf__learning_rate": [0.05],
                "clf__min_child_weight": [1],
                "clf__subsample": [0.85],
                "clf__colsample_bytree": [0.85],
                "clf__reg_lambda": [1.0],
            }
            if quick
            else {
                "clf__n_estimators": [250, 450],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.03, 0.08],
                "clf__min_child_weight": [1],
                "clf__subsample": [0.85],
                "clf__colsample_bytree": [0.85],
                "clf__reg_lambda": [1.0],
            }
        )
        scaler = "passthrough"
    else:  # pragma: no cover - protected by argparse choices
        raise SimulationError(f"Unknown model: {model_name}")

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                    keep_empty_features=True,
                ),
            ),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("scaler", scaler),
            ("clf", classifier),
        ]
    )
    return pipeline, grid


def valid_inner_splits(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, requested_splits: int, seed: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    unique_groups = np.unique(groups)
    maximum = min(requested_splits, len(unique_groups))
    for split_count in range(maximum, 1, -1):
        strategies = (
            (
                "StratifiedGroupKFold",
                StratifiedGroupKFold(
                    n_splits=split_count, shuffle=True, random_state=seed
                ),
            ),
            ("GroupKFold", GroupKFold(n_splits=split_count)),
        )
        for name, splitter in strategies:
            try:
                splits = list(splitter.split(X, y, groups))
            except ValueError:
                continue
            if all(
                np.unique(y[train]).size == 2
                and np.unique(y[validation]).size == 2
                for train, validation in splits
            ):
                return splits, name, split_count
    raise SimulationError(
        "Could not construct grouped inner CV whose every training and validation "
        "split has both classes"
    )


def prediction_scores(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        classes = np.asarray(estimator.classes_)
        location = np.flatnonzero(classes == 1)
        if len(location):
            return np.asarray(probabilities[:, location[0]], dtype=float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=float)
    return np.full(len(X), np.nan, dtype=float)


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_score: Sequence[float] | None = None) -> dict[str, float | int]:
    true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(y_pred, dtype=int)
    has_both_classes = np.array_equal(np.unique(true), np.array([0, 1]))
    balanced_accuracy = strict_binary_balanced_accuracy(true, predicted)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    result: dict[str, float | int] = {
        "n_samples": int(len(true)),
        "n_low": int((true == 0).sum()),
        "n_high": int((true == 1).sum()),
        "fold_has_both_classes": bool(has_both_classes),
        "accuracy": float(accuracy_score(true, predicted)),
        # Retain the established column name for output compatibility. Unlike the
        # previous implementation, an unobservable class makes fold-level BA NaN
        # instead of silently assigning that class a recall of zero.
        "balanced_accuracy_fixed": balanced_accuracy,
        "macro_precision_fixed": float(np.mean(precision)),
        "macro_recall_fixed": float(np.mean(recall)),
        "macro_f1_fixed": float(np.mean(f1)),
        "low_precision": float(precision[0]),
        "low_recall": float(recall[0]),
        "low_f1": float(f1[0]),
        "low_support": int(support[0]),
        "high_precision": float(precision[1]),
        "high_recall": float(recall[1]),
        "high_f1": float(f1[1]),
        "high_support": int(support[1]),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
        "roc_auc": math.nan,
        "average_precision": math.nan,
    }
    if y_score is not None:
        score = np.asarray(y_score, dtype=float)
        finite = np.isfinite(score)
        if finite.all() and np.unique(true).size == 2:
            result["roc_auc"] = float(roc_auc_score(true, score))
            result["average_precision"] = float(average_precision_score(true, score))
    return result


def bootstrap_confidence_intervals(predictions: pd.DataFrame, repeats: int, seed: int) -> dict[str, float | int]:
    metrics = (
        "accuracy",
        "balanced_accuracy_fixed",
        "macro_f1_fixed",
        "mcc",
        "roc_auc",
        "average_precision",
    )
    output: dict[str, float | int] = {"bootstrap_repeats": repeats}
    if repeats == 0:
        for metric in metrics:
            output[f"{metric}_ci_low"] = math.nan
            output[f"{metric}_ci_high"] = math.nan
        return output
    subjects = predictions["subject_id"].drop_duplicates().to_numpy()
    by_subject = {
        subject: predictions.index[predictions["subject_id"] == subject].to_numpy()
        for subject in subjects
    }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(repeats):
        drawn = rng.choice(subjects, size=len(subjects), replace=True)
        indices = np.concatenate([by_subject[subject] for subject in drawn])
        frame = predictions.loc[indices]
        values = classification_metrics(frame["y_true"], frame["y_pred"], frame["y_score"])
        for metric in metrics:
            value = float(values[metric])
            if math.isfinite(value):
                samples[metric].append(value)
    for metric in metrics:
        values = samples[metric]
        output[f"{metric}_ci_low"] = (
            float(np.percentile(values, 2.5)) if values else math.nan
        )
        output[f"{metric}_ci_high"] = (
            float(np.percentile(values, 97.5)) if values else math.nan
        )
    return output


def deterministic_subsample_indices(
    row_count: int, maximum_rows: int, seed: int
) -> np.ndarray:
    if maximum_rows <= 0 or row_count <= maximum_rows:
        return np.arange(row_count, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(row_count, size=maximum_rows, replace=False))


def positive_class_shap_values(
    values: Any, sample_count: int, feature_count: int
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (sample_count, feature_count):
        return array
    if array.ndim == 3 and array.shape[:2] == (sample_count, feature_count):
        if array.shape[2] != 2:
            raise SimulationError(
                f"Expected two SHAP outputs for binary classification, got {array.shape}"
            )
        return array[:, :, 1]
    if array.ndim == 3 and array.shape[0] == 2:
        candidate = array[1]
        if candidate.shape == (sample_count, feature_count):
            return candidate
    raise SimulationError(
        "Unexpected SHAP value shape: "
        f"{array.shape}; expected ({sample_count}, {feature_count})"
    )


def positive_class_base_values(base_values: Any, sample_count: int) -> np.ndarray:
    array = np.asarray(base_values, dtype=float)
    if array.ndim == 0:
        return np.full(sample_count, float(array))
    if array.shape == (sample_count,):
        return array
    if array.shape == (2,):
        return np.full(sample_count, float(array[1]))
    if array.shape == (sample_count, 2):
        return array[:, 1]
    if array.shape == (sample_count, 1):
        return array[:, 0]
    raise SimulationError(f"Unexpected SHAP base-value shape: {array.shape}")


def positive_class_model_output(estimator: Any, X: Any) -> tuple[np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise SimulationError(
                f"Expected binary predict_proba output, got {probabilities.shape}"
            )
        return probabilities[:, 1], "probability_class_1"
    if hasattr(estimator, "decision_function"):
        return (
            np.asarray(estimator.decision_function(X), dtype=float).reshape(-1),
            "decision_function",
        )
    return np.asarray(estimator.predict(X), dtype=float).reshape(-1), "predicted_class"


def tree_shap_explanation(
    estimator: Pipeline,
    X_background: pd.DataFrame,
    X_explained: pd.DataFrame,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    preprocessor = estimator[:-1]
    background = np.asarray(preprocessor.transform(X_background), dtype=float)
    explained = np.asarray(preprocessor.transform(X_explained), dtype=float)
    support = np.asarray(
        estimator.named_steps["variance"].get_support(), dtype=bool
    )
    retained_features = [
        feature for feature, retained in zip(features, support) if retained
    ]
    classifier = estimator.named_steps["clf"]
    explainer = shap.TreeExplainer(
        classifier,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
        feature_names=retained_features,
    )
    explanation = explainer(explained, check_additivity=False)
    retained_values = positive_class_shap_values(
        explanation.values,
        len(X_explained),
        len(retained_features),
    )
    values = np.zeros((len(X_explained), len(features)), dtype=float)
    values[:, support] = retained_values
    base_values = positive_class_base_values(
        explanation.base_values, len(X_explained)
    )
    model_output, output_name = positive_class_model_output(classifier, explained)
    return values, base_values, model_output, "TreeExplainer_interventional", output_name


def model_agnostic_shap_explanation(
    estimator: Pipeline,
    X_background: pd.DataFrame,
    X_explained: pd.DataFrame,
    features: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    background = X_background.to_numpy(dtype=float)
    explained = X_explained.to_numpy(dtype=float)

    def predict(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=features)
        output, _ = positive_class_model_output(estimator, frame)
        return output

    explainer = shap.Explainer(
        predict,
        background,
        algorithm="permutation",
        feature_names=list(features),
        seed=seed,
    )
    explanation = explainer(
        explained,
        max_evals=max(500, 2 * len(features) + 1),
        silent=True,
    )
    values = positive_class_shap_values(
        explanation.values, len(X_explained), len(features)
    )
    base_values = positive_class_base_values(
        explanation.base_values, len(X_explained)
    )
    model_output, output_name = positive_class_model_output(estimator, X_explained)
    return values, base_values, model_output, "PermutationExplainer", output_name


def validate_shap_additivity(
    values: np.ndarray,
    base_values: np.ndarray,
    model_output: np.ndarray,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, float]:
    reconstructed = base_values + values.sum(axis=1)
    additivity_errors = np.abs(reconstructed - model_output)
    maximum_additivity_error = float(additivity_errors.max(initial=0.0))
    if not np.isfinite(maximum_additivity_error) or maximum_additivity_error > 1e-3:
        raise SimulationError(
            f"{metadata['experiment_id']}, fold {metadata['fold']}: "
            "SHAP additivity check failed; maximum absolute error="
            f"{maximum_additivity_error:.6g}"
        )
    return reconstructed, additivity_errors, maximum_additivity_error


def summarize_shap_fold(
    values: np.ndarray,
    features: Sequence[str],
    metadata: dict[str, Any],
    explainer_name: str,
    output_name: str,
    background_samples: int,
    test_samples_available: int,
    top_k: int,
    maximum_additivity_error: float,
    elapsed: float,
) -> list[dict[str, Any]]:
    absolute_values = np.abs(values)
    mean_absolute = absolute_values.mean(axis=0)
    ranks = (
        pd.Series(mean_absolute)
        .rank(method="min", ascending=False)
        .astype(int)
        .to_numpy()
    )
    fold_rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        feature_values = values[:, feature_index]
        source_feature = feature.removesuffix("__mean").removesuffix("__std")
        fold_rows.append(
            {
                **metadata,
                "feature": feature,
                "source_feature": source_feature,
                "family": source_feature.split("_", 1)[0],
                "explainer": explainer_name,
                "model_output": output_name,
                "background_samples": background_samples,
                "test_samples_available": test_samples_available,
                "test_samples_explained": len(values),
                "mean_abs_shap": float(mean_absolute[feature_index]),
                "std_abs_shap_within_subject": float(
                    absolute_values[:, feature_index].std(ddof=0)
                ),
                "median_abs_shap": float(
                    np.median(absolute_values[:, feature_index])
                ),
                "max_abs_shap": float(absolute_values[:, feature_index].max()),
                "mean_signed_shap": float(feature_values.mean()),
                "positive_sample_fraction": float((feature_values > 0).mean()),
                "rank_within_fold": int(ranks[feature_index]),
                "top_k_within_fold": bool(ranks[feature_index] <= top_k),
                "top_k": top_k,
                "max_additivity_error": maximum_additivity_error,
                "shap_seconds_for_fold": elapsed,
            }
        )
    return fold_rows


def local_shap_rows(
    values: np.ndarray,
    base_values: np.ndarray,
    model_output: np.ndarray,
    reconstructed: np.ndarray,
    additivity_errors: np.ndarray,
    X_explained: pd.DataFrame,
    explained_frame: pd.DataFrame,
    explained_indices: np.ndarray,
    y_test: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    features: Sequence[str],
    metadata: dict[str, Any],
    explainer_name: str,
    output_name: str,
) -> list[dict[str, Any]]:
    local_rows: list[dict[str, Any]] = []
    raw_values = X_explained.to_numpy(dtype=float)
    for sample_position, (_, sample) in enumerate(explained_frame.iterrows()):
        original_position = explained_indices[sample_position]
        sample_metadata = {
            **metadata,
            "sample_id": sample["sample_id"],
            "trial_id": sample["trial_id"],
            "window_id": sample["window_id"],
            "emotion": sample["emotion"],
            "target_sam": int(sample["target_sam"]),
            "y_true": int(y_test[original_position]),
            "y_pred": int(predicted[original_position]),
            "y_score": float(scores[original_position]),
            "base_value": float(base_values[sample_position]),
            "explained_model_output": float(model_output[sample_position]),
            "reconstructed_model_output": float(reconstructed[sample_position]),
            "additivity_error": float(additivity_errors[sample_position]),
            "explainer": explainer_name,
            "model_output": output_name,
        }
        for feature_index, feature in enumerate(features):
            source_feature = feature.removesuffix("__mean").removesuffix("__std")
            local_rows.append(
                {
                    **sample_metadata,
                    "feature": feature,
                    "source_feature": source_feature,
                    "family": source_feature.split("_", 1)[0],
                    "raw_feature_value": float(
                        raw_values[sample_position, feature_index]
                    ),
                    "shap_value": float(values[sample_position, feature_index]),
                    "abs_shap_value": float(abs(values[sample_position, feature_index])),
                }
            )
    return local_rows


def shap_fold_rows(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_test: np.ndarray,
    predicted: np.ndarray,
    scores: np.ndarray,
    features: Sequence[str],
    metadata: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    background_indices = deterministic_subsample_indices(
        len(X_train), args.shap_background_size, stable_seed(seed, "background")
    )
    explained_indices = deterministic_subsample_indices(
        len(X_test), args.shap_max_test_samples, stable_seed(seed, "explained")
    )
    X_background = X_train.iloc[background_indices]
    X_explained = X_test.iloc[explained_indices]
    explained_frame = test_frame.iloc[explained_indices]
    shap_started = time.perf_counter()
    if metadata["model"] in {"rf", "xgboost"}:
        explanation = tree_shap_explanation(
            estimator, X_background, X_explained, features
        )
    else:
        explanation = model_agnostic_shap_explanation(
            estimator, X_background, X_explained, features, seed
        )
    values, base_values, model_output, explainer_name, output_name = explanation
    reconstructed, errors, maximum_error = validate_shap_additivity(
        values, base_values, model_output, metadata
    )
    elapsed = time.perf_counter() - shap_started
    fold_rows = summarize_shap_fold(
        values,
        features,
        metadata,
        explainer_name,
        output_name,
        len(background_indices),
        len(X_test),
        args.shap_top_k,
        maximum_error,
        elapsed,
    )
    local_rows = (
        local_shap_rows(
            values,
            base_values,
            model_output,
            reconstructed,
            errors,
            X_explained,
            explained_frame,
            explained_indices,
            y_test,
            predicted,
            scores,
            features,
            metadata,
            explainer_name,
            output_name,
        )
        if args.save_shap_values
        else []
    )
    return fold_rows, local_rows


class GCNBinaryClassifier(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = dropout
        self.output_layer = nn.Linear(hidden_channels, 2)

    def forward(self, data: Data) -> torch.Tensor:
        residual = self.input_projection(data.x)
        message = torch_f.relu(self.conv1(data.x, data.edge_index))
        values = self.norm1(residual + message)
        values = torch_f.dropout(
            values, p=self.dropout, training=self.training
        )
        message = torch_f.relu(self.conv2(values, data.edge_index))
        values = self.norm2(values + message)
        values = global_mean_pool(values, data.batch)
        return self.output_layer(values)


class GATBinaryClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.dropout = dropout
        self.output_layer = nn.Linear(hidden_channels, 2)

    def forward(self, data: Data) -> torch.Tensor:
        residual = self.input_projection(data.x)
        message = torch_f.elu(self.gat1(data.x, data.edge_index))
        values = self.norm1(residual + message)
        values = torch_f.dropout(
            values, p=self.dropout, training=self.training
        )
        message = torch_f.elu(self.gat2(values, data.edge_index))
        values = self.norm2(values + message)
        values = global_mean_pool(values, data.batch)
        return self.output_layer(values)


def set_graph_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_graph_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SimulationError(
                "--graph-device cuda was requested, but CUDA is unavailable"
            )
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def graph_parameter_grid(
    model_name: str, quick: bool
) -> list[dict[str, Any]]:
    if model_name == "gcn":
        grid: dict[str, list[Any]] = {
            "hidden_channels": [32, 64],
            "dropout": [0.2, 0.5],
            "learning_rate": [0.001, 0.0003, 0.003],
            "weight_decay": [0.0, 5e-4],
        }
    elif model_name == "gat":
        grid = {
            "hidden_channels": [32, 64],
            "heads": [2, 4],
            "dropout": [0.2, 0.5],
            "learning_rate": [0.001, 0.0003, 0.003],
            "weight_decay": [0.0, 5e-4],
        }
    else:  # pragma: no cover - protected by dispatcher
        raise SimulationError(f"Unknown graph model: {model_name}")
    if quick:
        grid = {name: [values[0]] for name, values in grid.items()}
    return [dict(parameters) for parameters in ParameterGrid(grid)]


def graph_column_groups(
    features: Sequence[str], representation: str
) -> list[list[str]]:
    if representation not in GRAPH_REPRESENTATIONS:
        raise SimulationError(
            f"GCN/GAT do not support representation {representation!r}; "
            f"use one of {GRAPH_REPRESENTATIONS}"
        )
    groups: list[list[str]] = []
    expected_suffixes: list[str] | None = None
    for modality in ("bvp", "eda", "temp"):
        prefix = f"{representation}_{modality}_"
        suffix_to_column = {
            feature[len(prefix) :]: feature
            for feature in features
            if feature.startswith(prefix)
        }
        suffixes = sorted(suffix_to_column)
        if not suffixes:
            raise SimulationError(
                f"No {representation.upper()} columns found for {modality.upper()}"
            )
        if expected_suffixes is None:
            expected_suffixes = suffixes
        elif suffixes != expected_suffixes:
            raise SimulationError(
                f"{representation.upper()} graph-node feature schema differs "
                f"between modalities; {modality.upper()} cannot be aligned"
            )
        groups.append([suffix_to_column[suffix] for suffix in suffixes])
    return groups


def fit_graph_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(
        strategy="median", add_indicator=False, keep_empty_features=True
    )
    scaler = StandardScaler()
    imputed = imputer.fit_transform(X_train)
    scaler.fit(imputed)
    return imputer, scaler


def transform_graph_features(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> pd.DataFrame:
    transformed = scaler.transform(imputer.transform(X))
    return pd.DataFrame(transformed, columns=X.columns, index=X.index)


def modality_edge_index(node_count: int) -> torch.Tensor:
    edges = [
        (source, target)
        for source in range(node_count)
        for target in range(node_count)
        if source != target
    ]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def graph_samples(
    X: pd.DataFrame,
    y: Sequence[int],
    column_groups: Sequence[Sequence[str]],
) -> list[Data]:
    labels = np.asarray(y, dtype=int)
    if len(X) != len(labels):
        raise SimulationError("Graph feature/label lengths do not match")
    node_count = len(column_groups)
    identity = np.eye(node_count, dtype=np.float32)
    edge_index = modality_edge_index(node_count)
    graphs: list[Data] = []
    for position in range(len(X)):
        row = X.iloc[position]
        nodes = []
        for node_index, columns in enumerate(column_groups):
            measurements = row[list(columns)].to_numpy(dtype=np.float32)
            nodes.append(np.concatenate([measurements, identity[node_index]]))
        graphs.append(
            Data(
                x=torch.from_numpy(np.stack(nodes)),
                edge_index=edge_index.clone(),
                y=torch.tensor([int(labels[position])], dtype=torch.long),
            )
        )
    return graphs


def make_graph_model(
    model_name: str, in_channels: int, parameters: dict[str, Any]
) -> nn.Module:
    common = {
        "in_channels": in_channels,
        "hidden_channels": int(parameters["hidden_channels"]),
        "dropout": float(parameters["dropout"]),
    }
    if model_name == "gcn":
        return GCNBinaryClassifier(**common)
    if model_name == "gat":
        return GATBinaryClassifier(
            **common, heads=int(parameters["heads"])
        )
    raise SimulationError(f"Unknown graph model: {model_name}")


def graph_class_weights(
    labels: Sequence[int], device: torch.device
) -> torch.Tensor:
    values = np.asarray(labels, dtype=int)
    counts = np.bincount(values, minlength=2)
    if np.any(counts == 0):
        raise SimulationError("Graph training data must contain both classes")
    weights = len(values) / (2.0 * counts.astype(float))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def graph_loader(
    graphs: Sequence[Data], batch_size: int, shuffle: bool
) -> GraphDataLoader:
    return GraphDataLoader(
        list(graphs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def graph_validation_loss(
    model: nn.Module,
    loader: GraphDataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch.y.reshape(-1))
            count = int(batch.num_graphs)
            total += float(loss.item()) * count
            samples += count
    if not samples:
        raise SimulationError("Graph validation loader is empty")
    return total / samples


def train_graph_network(
    model: nn.Module,
    train_loader: GraphDataLoader,
    train_labels: Sequence[int],
    parameters: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    validation_loader: GraphDataLoader | None = None,
    fixed_epochs: int | None = None,
) -> int:
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=graph_class_weights(train_labels, device)
    )
    optimizer = Adam(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    maximum_epochs = fixed_epochs or args.graph_max_epochs
    best_epoch = maximum_epochs
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch.y.reshape(-1))
            loss.backward()
            optimizer.step()

        if validation_loader is None:
            continue
        validation_loss = graph_validation_loss(
            model, validation_loader, criterion, device
        )
        if epoch < args.graph_min_epochs:
            continue
        if validation_loss < best_loss - args.graph_min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.graph_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return max(1, best_epoch)


def predict_graph_network(
    model: nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_values: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probabilities = torch.softmax(model(batch), dim=1)
            predicted = probabilities.argmax(dim=1)
            true_values.extend(
                batch.y.detach().cpu().numpy().reshape(-1).astype(int).tolist()
            )
            predictions.extend(
                predicted.detach().cpu().numpy().astype(int).tolist()
            )
            scores.extend(
                probabilities[:, 1].detach().cpu().numpy().astype(float).tolist()
            )
    return (
        np.asarray(true_values, dtype=int),
        np.asarray(predictions, dtype=int),
        np.asarray(scores, dtype=float),
    )


def select_graph_parameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    column_groups: Sequence[Sequence[str]],
    model_name: str,
    args: argparse.Namespace,
    fold_seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], float, int, str, int]:
    inner_splits, inner_name, actual_splits = valid_inner_splits(
        X_train, y_train, groups_train, args.inner_splits, fold_seed
    )
    best_parameters: dict[str, Any] | None = None
    best_score = -math.inf
    best_epochs: list[int] = []

    for parameter_index, parameters in enumerate(
        graph_parameter_grid(model_name, args.quick)
    ):
        validation_scores: list[float] = []
        parameter_epochs: list[int] = []
        for inner_fold, (fit_index, validation_index) in enumerate(
            inner_splits, start=1
        ):
            inner_seed = stable_seed(
                fold_seed, model_name, parameter_index, inner_fold
            )
            set_graph_seed(inner_seed)
            fit_frame = X_train.iloc[fit_index]
            validation_frame = X_train.iloc[validation_index]
            imputer, scaler = fit_graph_preprocessor(fit_frame)
            fit_transformed = transform_graph_features(
                fit_frame, imputer, scaler
            )
            validation_transformed = transform_graph_features(
                validation_frame, imputer, scaler
            )
            fit_graphs = graph_samples(
                fit_transformed, y_train[fit_index], column_groups
            )
            validation_graphs = graph_samples(
                validation_transformed,
                y_train[validation_index],
                column_groups,
            )
            fit_loader = graph_loader(
                fit_graphs, args.graph_batch_size, True
            )
            validation_loader = graph_loader(
                validation_graphs, args.graph_eval_batch_size, False
            )
            in_channels = fit_graphs[0].x.shape[1]
            model = make_graph_model(
                model_name, int(in_channels), parameters
            )
            best_epoch = train_graph_network(
                model,
                fit_loader,
                y_train[fit_index],
                parameters,
                args,
                device,
                validation_loader=validation_loader,
            )
            validation_true, validation_predicted, _ = (
                predict_graph_network(model, validation_loader, device)
            )
            validation_scores.append(
                strict_binary_balanced_accuracy(
                    validation_true, validation_predicted
                )
            )
            parameter_epochs.append(best_epoch)

        mean_score = float(np.mean(validation_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_parameters = dict(parameters)
            best_epochs = parameter_epochs

    if best_parameters is None or not np.isfinite(best_score):
        raise SimulationError(
            f"Graph hyperparameter selection failed for {model_name}"
        )
    refit_epochs = max(1, int(round(float(np.median(best_epochs)))))
    return (
        best_parameters,
        best_score,
        refit_epochs,
        inner_name,
        actual_splits,
    )


def run_graph_experiment(
    item: InputDataset,
    target: str,
    input_mode: str,
    representation: str,
    model_name: str,
    args: argparse.Namespace,
    experiment_index: int,
    experiment_total: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    raw_features = representation_columns(item, representation)
    frame, features = target_table(item, target, input_mode, raw_features)
    X = frame[features].astype(float)
    y = frame["y"].to_numpy(dtype=int)
    groups = frame["subject_id"].to_numpy()
    subjects = pd.unique(groups)
    column_groups = graph_column_groups(features, representation)
    experiment_id = safe_token(
        f"{item.input_id}__{target}__{input_mode}__{representation}__{model_name}"
    )
    print(
        f"[{experiment_index:03d}/{experiment_total:03d}] {experiment_id}: "
        f"{len(frame):,} graphs, {len(subjects)} subjects, "
        f"{len(column_groups)} nodes, {len(column_groups[0]) + len(column_groups)} "
        "attributes/node",
        flush=True,
    )
    logo_splits = list(LeaveOneGroupOut().split(X, y, groups))
    if args.max_outer_folds:
        logo_splits = logo_splits[: args.max_outer_folds]
    complete_loso = len(logo_splits) == len(subjects)
    experiment_started = time.perf_counter()
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    device = resolve_graph_device(args.graph_device)

    for fold_number, (train_index, test_index) in enumerate(
        logo_splits, start=1
    ):
        fold_started = time.perf_counter()
        held_out = pd.unique(groups[test_index])
        if len(held_out) != 1:
            raise SimulationError(
                f"{experiment_id}: LOSO fold has {len(held_out)} test subjects"
            )
        fold_seed = stable_seed(args.seed, experiment_id, fold_number)
        fit_started = time.perf_counter()
        (
            best_parameters,
            best_inner_score,
            refit_epochs,
            inner_name,
            actual_inner_splits,
        ) = select_graph_parameters(
            X.iloc[train_index],
            y[train_index],
            groups[train_index],
            column_groups,
            model_name,
            args,
            fold_seed,
            device,
        )

        set_graph_seed(stable_seed(fold_seed, "outer_refit"))
        train_frame = X.iloc[train_index]
        test_features = X.iloc[test_index]
        imputer, scaler = fit_graph_preprocessor(train_frame)
        train_transformed = transform_graph_features(
            train_frame, imputer, scaler
        )
        test_transformed = transform_graph_features(
            test_features, imputer, scaler
        )
        train_graphs = graph_samples(
            train_transformed, y[train_index], column_groups
        )
        test_graphs = graph_samples(
            test_transformed, y[test_index], column_groups
        )
        train_loader = graph_loader(
            train_graphs, args.graph_batch_size, True
        )
        test_loader = graph_loader(
            test_graphs, args.graph_eval_batch_size, False
        )
        model = make_graph_model(
            model_name, int(train_graphs[0].x.shape[1]), best_parameters
        )
        train_graph_network(
            model,
            train_loader,
            y[train_index],
            best_parameters,
            args,
            device,
            fixed_epochs=refit_epochs,
        )
        fit_seconds = time.perf_counter() - fit_started
        predict_started = time.perf_counter()
        predicted_true, predicted, scores = predict_graph_network(
            model, test_loader, device
        )
        predict_seconds = time.perf_counter() - predict_started
        if not np.array_equal(predicted_true, y[test_index]):
            raise SimulationError(
                f"{experiment_id}: graph loader changed test sample order"
            )

        test_frame = frame.iloc[test_index]
        for position, (_, sample) in enumerate(test_frame.iterrows()):
            prediction_rows.append(
                {
                    "experiment_id": experiment_id,
                    "input_id": item.input_id,
                    "window_sec": item.window_sec,
                    "overlap_sec": item.overlap_sec,
                    "target": target,
                    "input_mode": input_mode,
                    "representation": representation,
                    "model": model_name,
                    "fold": fold_number,
                    "subject_id": sample["subject_id"],
                    "emotion": sample["emotion"],
                    "movie_order_index": sample["movie_order_index"],
                    "trial_id": sample["trial_id"],
                    "sample_id": sample["sample_id"],
                    "window_id": sample["window_id"],
                    "target_sam": int(sample["target_sam"]),
                    "y_true": int(y[test_index][position]),
                    "y_pred": int(predicted[position]),
                    "y_score": float(scores[position]),
                }
            )
        fold_metrics = classification_metrics(
            y[test_index], predicted, scores
        )
        reported_parameters = {
            **best_parameters,
            "refit_epochs": refit_epochs,
        }
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "input_id": item.input_id,
                "window_sec": item.window_sec,
                "overlap_sec": item.overlap_sec,
                "target": target,
                "input_mode": input_mode,
                "representation": representation,
                "model": model_name,
                "fold": fold_number,
                "subject_id": held_out[0],
                "outer_train_samples": len(train_index),
                "outer_test_samples": len(test_index),
                "outer_train_subjects": len(np.unique(groups[train_index])),
                "inner_cv": inner_name,
                "inner_splits": actual_inner_splits,
                "best_inner_balanced_accuracy_fixed": best_inner_score,
                "best_parameters_json": json.dumps(
                    reported_parameters, sort_keys=True
                ),
                "input_feature_count": len(features),
                "retained_feature_count": len(features),
                "zero_variance_removed_count": 0,
                "graph_node_count": len(column_groups),
                "graph_features_per_node": (
                    len(column_groups[0]) + len(column_groups)
                ),
                "graph_device": str(device),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fold_seconds": time.perf_counter() - fold_started,
                **fold_metrics,
            }
        )
        print(
            f"    fold {fold_number:02d}/{len(logo_splits):02d}, "
            f"subject {held_out[0]}: "
            f"balanced-accuracy="
            f"{float(fold_metrics['balanced_accuracy_fixed']):.3f}, "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    pooled = classification_metrics(
        predictions["y_true"], predictions["y_pred"], predictions["y_score"]
    )
    intervals = bootstrap_confidence_intervals(
        predictions,
        args.bootstrap_repeats,
        stable_seed(args.seed, experiment_id, "bootstrap"),
    )
    summary = {
        "experiment_id": experiment_id,
        "input_id": item.input_id,
        "window_sec": item.window_sec,
        "overlap_sec": item.overlap_sec,
        "target": target,
        "input_mode": input_mode,
        "representation": representation,
        "model": model_name,
        "feature_count_before_aggregation": len(raw_features),
        "input_feature_count": len(features),
        "subjects_total": len(subjects),
        "outer_folds_run": len(logo_splits),
        "complete_loso": complete_loso,
        "quick_grid": bool(args.quick),
        "experiment_seconds": time.perf_counter() - experiment_started,
        **pooled,
        **intervals,
    }
    return summary, fold_rows, prediction_rows, [], [], []


def run_experiment(
    item: InputDataset,
    target: str,
    input_mode: str,
    representation: str,
    model_name: str,
    args: argparse.Namespace,
    experiment_index: int,
    experiment_total: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if model_name in GRAPH_MODEL_NAMES:
        return run_graph_experiment(
            item,
            target,
            input_mode,
            representation,
            model_name,
            args,
            experiment_index,
            experiment_total,
        )
    raw_features = representation_columns(item, representation)
    frame, features = target_table(item, target, input_mode, raw_features)
    X = frame[features].astype(float)
    y = frame["y"].to_numpy(dtype=int)
    groups = frame["subject_id"].to_numpy()
    subjects = pd.unique(groups)
    experiment_id = safe_token(
        f"{item.input_id}__{target}__{input_mode}__{representation}__{model_name}"
    )
    print(
        f"[{experiment_index:03d}/{experiment_total:03d}] {experiment_id}: "
        f"{len(frame):,} samples, {len(subjects)} subjects, {len(features)} input columns",
        flush=True,
    )
    logo_splits = list(LeaveOneGroupOut().split(X, y, groups))
    if args.max_outer_folds:
        logo_splits = logo_splits[: args.max_outer_folds]
    complete_loso = len(logo_splits) == len(subjects)
    experiment_started = time.perf_counter()
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    shap_importance_rows: list[dict[str, Any]] = []
    shap_value_rows: list[dict[str, Any]] = []

    for fold_number, (train_index, test_index) in enumerate(logo_splits, start=1):
        fold_started = time.perf_counter()
        held_out = pd.unique(groups[test_index])
        if len(held_out) != 1:
            raise SimulationError(f"{experiment_id}: LOSO fold has {len(held_out)} test subjects")
        fold_seed = stable_seed(args.seed, experiment_id, fold_number)
        pipeline, grid = model_spec(
            model_name, fold_seed, args.n_jobs, args.device, args.quick
        )
        inner_splits, inner_name, actual_inner_splits = valid_inner_splits(
            X.iloc[train_index],
            y[train_index],
            groups[train_index],
            args.inner_splits,
            fold_seed,
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=STRICT_BALANCED_ACCURACY,
            cv=inner_splits,
            n_jobs=args.n_jobs,
            refit=True,
            error_score="raise",
            return_train_score=False,
        )
        fit_started = time.perf_counter()
        search.fit(
            X.iloc[train_index], y[train_index], groups=groups[train_index]
        )
        fit_seconds = time.perf_counter() - fit_started
        estimator = search.best_estimator_
        check_is_fitted(estimator)
        predict_started = time.perf_counter()
        predicted = estimator.predict(X.iloc[test_index]).astype(int)
        scores = prediction_scores(estimator, X.iloc[test_index])
        predict_seconds = time.perf_counter() - predict_started
        test_frame = frame.iloc[test_index]
        for position, (_, sample) in enumerate(test_frame.iterrows()):
            prediction_rows.append(
                {
                    "experiment_id": experiment_id,
                    "input_id": item.input_id,
                    "window_sec": item.window_sec,
                    "overlap_sec": item.overlap_sec,
                    "target": target,
                    "input_mode": input_mode,
                    "representation": representation,
                    "model": model_name,
                    "fold": fold_number,
                    "subject_id": sample["subject_id"],
                    "emotion": sample["emotion"],
                    "movie_order_index": sample["movie_order_index"],
                    "trial_id": sample["trial_id"],
                    "sample_id": sample["sample_id"],
                    "window_id": sample["window_id"],
                    "target_sam": int(sample["target_sam"]),
                    "y_true": int(y[test_index][position]),
                    "y_pred": int(predicted[position]),
                    "y_score": float(scores[position]),
                }
            )
        fold_metrics = classification_metrics(y[test_index], predicted, scores)
        variance_support = estimator.named_steps["variance"].get_support()
        fold_rows.append(
            {
                "experiment_id": experiment_id,
                "input_id": item.input_id,
                "window_sec": item.window_sec,
                "overlap_sec": item.overlap_sec,
                "target": target,
                "input_mode": input_mode,
                "representation": representation,
                "model": model_name,
                "fold": fold_number,
                "subject_id": held_out[0],
                "outer_train_samples": len(train_index),
                "outer_test_samples": len(test_index),
                "outer_train_subjects": len(np.unique(groups[train_index])),
                "inner_cv": inner_name,
                "inner_splits": actual_inner_splits,
                "best_inner_balanced_accuracy_fixed": float(search.best_score_),
                "best_parameters_json": json.dumps(search.best_params_, sort_keys=True),
                "input_feature_count": len(features),
                "retained_feature_count": int(variance_support.sum()),
                "zero_variance_removed_count": int((~variance_support).sum()),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "fold_seconds": time.perf_counter() - fold_started,
                **fold_metrics,
            }
        )

        if args.permutation_repeats and model_name in args.importance_models:
            importance_started = time.perf_counter()
            result = permutation_importance(
                estimator,
                X.iloc[test_index],
                y[test_index],
                scoring=STRICT_BALANCED_ACCURACY,
                n_repeats=args.permutation_repeats,
                random_state=fold_seed,
                n_jobs=args.n_jobs,
            )
            for feature, mean, std in zip(
                features, result.importances_mean, result.importances_std
            ):
                source_feature = feature.removesuffix("__mean").removesuffix("__std")
                importance_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "input_id": item.input_id,
                        "window_sec": item.window_sec,
                        "overlap_sec": item.overlap_sec,
                        "target": target,
                        "input_mode": input_mode,
                        "representation": representation,
                        "model": model_name,
                        "fold": fold_number,
                        "subject_id": held_out[0],
                        "feature": feature,
                        "source_feature": source_feature,
                        "family": source_feature.split("_", 1)[0],
                        "permutation_importance_mean": float(mean),
                        "permutation_importance_std": float(std),
                        "permutation_repeats": args.permutation_repeats,
                        "importance_seconds_for_fold": time.perf_counter() - importance_started,
                    }
                )
        if not args.no_shap and model_name in args.shap_models:
            fold_shap_rows, local_shap_rows = shap_fold_rows(
                estimator=estimator,
                X_train=X.iloc[train_index],
                X_test=X.iloc[test_index],
                test_frame=test_frame,
                y_test=y[test_index],
                predicted=predicted,
                scores=scores,
                features=features,
                metadata={
                    "experiment_id": experiment_id,
                    "input_id": item.input_id,
                    "window_sec": item.window_sec,
                    "overlap_sec": item.overlap_sec,
                    "target": target,
                    "input_mode": input_mode,
                    "representation": representation,
                    "model": model_name,
                    "fold": fold_number,
                    "subject_id": held_out[0],
                },
                args=args,
                seed=fold_seed,
            )
            shap_importance_rows.extend(fold_shap_rows)
            shap_value_rows.extend(local_shap_rows)
        print(
            f"    fold {fold_number:02d}/{len(logo_splits):02d}, subject {held_out[0]}: "
            f"balanced-accuracy={float(fold_metrics['balanced_accuracy_fixed']):.3f}, "
            f"fit={fit_seconds:.1f}s",
            flush=True,
        )

    predictions = pd.DataFrame(prediction_rows)
    pooled = classification_metrics(
        predictions["y_true"], predictions["y_pred"], predictions["y_score"]
    )
    intervals = bootstrap_confidence_intervals(
        predictions,
        args.bootstrap_repeats,
        stable_seed(args.seed, experiment_id, "bootstrap"),
    )
    summary = {
        "experiment_id": experiment_id,
        "input_id": item.input_id,
        "window_sec": item.window_sec,
        "overlap_sec": item.overlap_sec,
        "target": target,
        "input_mode": input_mode,
        "representation": representation,
        "model": model_name,
        "feature_count_before_aggregation": len(raw_features),
        "input_feature_count": len(features),
        "subjects_total": len(subjects),
        "outer_folds_run": len(logo_splits),
        "complete_loso": complete_loso,
        "quick_grid": bool(args.quick),
        "experiment_seconds": time.perf_counter() - experiment_started,
        **pooled,
        **intervals,
    }
    return (
        summary,
        fold_rows,
        prediction_rows,
        importance_rows,
        shap_importance_rows,
        shap_value_rows,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return adjusted.tolist()
    ordered = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def cluster_permutation_result(
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    metric: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    """Compare paired OOF predictions by swapping whole subject clusters."""
    keys = ["subject_id", "sample_id"]
    required = {*keys, "y_true", "y_pred"}
    missing = (required - set(left_frame.columns)) | (
        required - set(right_frame.columns)
    )
    if missing:
        raise SimulationError(
            "Paired cluster permutation is missing prediction columns: "
            + ", ".join(sorted(missing))
        )
    if left_frame.duplicated(keys).any() or right_frame.duplicated(keys).any():
        raise SimulationError(
            "Paired cluster permutation requires one OOF prediction per subject/sample"
        )

    paired = left_frame[[*keys, "y_true", "y_pred"]].merge(
        right_frame[[*keys, "y_true", "y_pred"]],
        on=keys,
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
        validate="one_to_one",
    )
    if not (paired["_merge"] == "both").all():
        return {
            "pairing_status": "incompatible_sample_sets",
            "paired_subjects": 0,
            "paired_samples": 0,
            "pooled_metric_a": math.nan,
            "pooled_metric_b": math.nan,
            "observed_difference_a_minus_b": math.nan,
            "permutation_extreme_count": math.nan,
            "p_value": math.nan,
        }
    if not np.array_equal(
        paired["y_true_a"].to_numpy(dtype=int),
        paired["y_true_b"].to_numpy(dtype=int),
    ):
        return {
            "pairing_status": "incompatible_true_labels",
            "paired_subjects": 0,
            "paired_samples": 0,
            "pooled_metric_a": math.nan,
            "pooled_metric_b": math.nan,
            "observed_difference_a_minus_b": math.nan,
            "permutation_extreme_count": math.nan,
            "p_value": math.nan,
        }

    true = paired["y_true_a"].to_numpy(dtype=int)
    predicted_a = paired["y_pred_a"].to_numpy(dtype=int)
    predicted_b = paired["y_pred_b"].to_numpy(dtype=int)
    correct_a = (predicted_a == true).astype(float)
    correct_b = (predicted_b == true).astype(float)
    correct_difference = correct_a - correct_b

    if metric == "accuracy":
        pooled_a = float(correct_a.mean())
        pooled_b = float(correct_b.mean())
        row_contribution = correct_difference / len(paired)
    elif metric == "balanced_accuracy_fixed":
        class_counts = np.bincount(true, minlength=2)
        if np.any(class_counts == 0):
            return {
                "pairing_status": "pooled_target_is_single_class",
                "paired_subjects": int(paired["subject_id"].nunique()),
                "paired_samples": int(len(paired)),
                "pooled_metric_a": math.nan,
                "pooled_metric_b": math.nan,
                "observed_difference_a_minus_b": math.nan,
                "permutation_extreme_count": math.nan,
                "p_value": math.nan,
            }
        class_weight = 0.5 / class_counts[true]
        pooled_a = float(np.sum(correct_a * class_weight))
        pooled_b = float(np.sum(correct_b * class_weight))
        row_contribution = correct_difference * class_weight
    else:
        raise SimulationError(f"Unsupported paired-test metric: {metric}")

    subject_contributions = (
        pd.DataFrame(
            {
                "subject_id": paired["subject_id"],
                "contribution": row_contribution,
            }
        )
        .groupby("subject_id", sort=False)["contribution"]
        .sum()
        .to_numpy(dtype=float)
    )
    observed_difference = float(subject_contributions.sum())
    rng = np.random.default_rng(seed)
    signs = rng.integers(
        0, 2, size=(repeats, len(subject_contributions)), dtype=np.int8
    )
    signs = signs * 2 - 1
    permuted_differences = signs @ subject_contributions
    tolerance = np.finfo(float).eps * max(1.0, abs(observed_difference)) * 8
    extreme_count = int(
        np.count_nonzero(
            np.abs(permuted_differences) + tolerance >= abs(observed_difference)
        )
    )
    p_value = float((extreme_count + 1) / (repeats + 1))
    return {
        "pairing_status": "ok",
        "paired_subjects": int(len(subject_contributions)),
        "paired_samples": int(len(paired)),
        "pooled_metric_a": pooled_a,
        "pooled_metric_b": pooled_b,
        "observed_difference_a_minus_b": observed_difference,
        "permutation_extreme_count": extreme_count,
        "p_value": p_value,
    }


def paired_tests(
    predictions: pd.DataFrame, repeats: int = 10_000, seed: int = 42
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []

    def compare(group: pd.DataFrame, comparison_type: str, family_id: str) -> None:
        experiments = group["experiment_id"].drop_duplicates().tolist()
        for left, right in combinations(experiments, 2):
            left_frame = group.loc[group["experiment_id"] == left]
            right_frame = group.loc[group["experiment_id"] == right]
            left_meta = left_frame.iloc[0]
            right_meta = right_frame.iloc[0]
            for metric in ("accuracy", "balanced_accuracy_fixed"):
                test_seed = stable_seed(
                    seed, family_id, left, right, metric, "subject_cluster_permutation"
                )
                if left_meta["input_id"] != right_meta["input_id"]:
                    test = {
                        "pairing_status": "different_input_settings_require_common_unit",
                        "paired_subjects": 0,
                        "paired_samples": 0,
                        "pooled_metric_a": math.nan,
                        "pooled_metric_b": math.nan,
                        "observed_difference_a_minus_b": math.nan,
                        "permutation_extreme_count": math.nan,
                        "p_value": math.nan,
                    }
                else:
                    test = cluster_permutation_result(
                        left_frame, right_frame, metric, repeats, test_seed
                    )
                rows.append(
                    {
                        "comparison_family": family_id,
                        "comparison_type": comparison_type,
                        "test_method": "paired_subject_cluster_permutation",
                        "metric": metric,
                        "experiment_a": left,
                        "experiment_b": right,
                        "input_a": left_meta["input_id"],
                        "input_b": right_meta["input_id"],
                        "target": left_meta["target"],
                        "input_mode": left_meta["input_mode"],
                        "representation_a": left_meta["representation"],
                        "representation_b": right_meta["representation"],
                        "model_a": left_meta["model"],
                        "model_b": right_meta["model"],
                        "permutation_repeats": repeats,
                        **test,
                    }
                )

    # Representations
    for keys, group in predictions.groupby(
        ["input_id", "target", "input_mode", "model"], sort=False
    ):
        if group["representation"].nunique() > 1:
            family = "representation|" + "|".join(map(str, keys))
            compare(group, "representations_within_model", family)
    # Models
    for keys, group in predictions.groupby(
        ["input_id", "target", "input_mode", "representation"], sort=False
    ):
        if group["model"].nunique() > 1:
            family = "model|" + "|".join(map(str, keys))
            compare(group, "models_within_representation", family)
    # Window and overlap settings. Tests are emitted only when sample identifiers
    # align exactly; otherwise pairing_status documents why no p-value is reported.
    for keys, group in predictions.groupby(
        ["target", "input_mode", "representation", "model"], sort=False
    ):
        if group["input_id"].nunique() > 1:
            family = "sensitivity|" + "|".join(map(str, keys))
            compare(group, "window_overlap_sensitivity", family)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["p_value_holm"] = math.nan
    for _, indices in result.groupby(
        ["comparison_family", "metric"], sort=False
    ).groups.items():
        adjusted = holm_adjust(result.loc[indices, "p_value"].to_numpy())
        result.loc[indices, "p_value_holm"] = adjusted
    result["significant_holm_0_05"] = result["p_value_holm"] < 0.05
    return result


def importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    if importance.empty:
        return pd.DataFrame()
    group_columns = [
        "experiment_id",
        "input_id",
        "window_sec",
        "overlap_sec",
        "target",
        "input_mode",
        "representation",
        "model",
        "feature",
        "source_feature",
        "family",
    ]
    summary = (
        importance.groupby(group_columns, dropna=False, sort=False)[
            "permutation_importance_mean"
        ]
        .agg(
            mean_importance="mean",
            std_importance_across_folds="std",
            median_importance="median",
            min_importance="min",
            max_importance="max",
            evaluated_folds="count",
        )
        .reset_index()
    )
    positive = (
        importance.assign(positive=importance["permutation_importance_mean"] > 0)
        .groupby(group_columns, dropna=False, sort=False)["positive"]
        .mean()
        .rename("positive_fold_fraction")
        .reset_index()
    )
    summary = summary.merge(positive, on=group_columns, how="left", validate="one_to_one")
    summary["rank_within_experiment"] = summary.groupby("experiment_id")[
        "mean_importance"
    ].rank(method="min", ascending=False)
    return summary.sort_values(
        ["experiment_id", "rank_within_experiment", "feature"], kind="stable"
    )


def shap_importance_summary(shap_importance: pd.DataFrame) -> pd.DataFrame:
    if shap_importance.empty:
        return pd.DataFrame()
    group_columns = [
        "experiment_id",
        "input_id",
        "window_sec",
        "overlap_sec",
        "target",
        "input_mode",
        "representation",
        "model",
        "feature",
        "source_feature",
        "family",
        "explainer",
        "model_output",
        "top_k",
    ]
    summary = (
        shap_importance.groupby(group_columns, dropna=False, sort=False)
        .agg(
            mean_abs_shap_across_subjects=("mean_abs_shap", "mean"),
            std_abs_shap_across_subjects=("mean_abs_shap", "std"),
            median_abs_shap_across_subjects=("mean_abs_shap", "median"),
            min_abs_shap_across_subjects=("mean_abs_shap", "min"),
            max_abs_shap_across_subjects=("mean_abs_shap", "max"),
            mean_signed_shap_across_subjects=("mean_signed_shap", "mean"),
            std_signed_shap_across_subjects=("mean_signed_shap", "std"),
            mean_rank_across_subjects=("rank_within_fold", "mean"),
            std_rank_across_subjects=("rank_within_fold", "std"),
            median_rank_across_subjects=("rank_within_fold", "median"),
            best_rank_across_subjects=("rank_within_fold", "min"),
            worst_rank_across_subjects=("rank_within_fold", "max"),
            top_k_subject_fraction=("top_k_within_fold", "mean"),
            positive_direction_subject_fraction=(
                "mean_signed_shap",
                lambda values: float((values > 0).mean()),
            ),
            evaluated_subjects=("subject_id", "nunique"),
            evaluated_folds=("fold", "nunique"),
            total_explained_samples=("test_samples_explained", "sum"),
            max_additivity_error=("max_additivity_error", "max"),
        )
        .reset_index()
    )
    mean_importance = summary["mean_abs_shap_across_subjects"].to_numpy()
    std_importance = summary["std_abs_shap_across_subjects"].fillna(0).to_numpy()
    summary["importance_cv_across_subjects"] = np.divide(
        std_importance,
        mean_importance,
        out=np.zeros_like(std_importance, dtype=float),
        where=mean_importance > 0,
    )
    positive = summary["positive_direction_subject_fraction"]
    summary["direction_consistency_subject_fraction"] = np.maximum(
        positive, 1.0 - positive
    )
    summary["rank_by_mean_abs_shap"] = summary.groupby("experiment_id")[
        "mean_abs_shap_across_subjects"
    ].rank(method="min", ascending=False)
    return summary.sort_values(
        ["experiment_id", "rank_by_mean_abs_shap", "feature"], kind="stable"
    )


def shap_family_summary(shap_importance: pd.DataFrame) -> pd.DataFrame:
    if shap_importance.empty:
        return pd.DataFrame()
    fold_columns = [
        "experiment_id",
        "input_id",
        "window_sec",
        "overlap_sec",
        "target",
        "input_mode",
        "representation",
        "model",
        "fold",
        "subject_id",
        "family",
    ]
    per_fold = (
        shap_importance.groupby(fold_columns, dropna=False, sort=False)
        .agg(
            family_mean_abs_shap=("mean_abs_shap", "sum"),
            family_feature_count=("feature", "nunique"),
        )
        .reset_index()
    )
    totals = per_fold.groupby(["experiment_id", "fold"], sort=False)[
        "family_mean_abs_shap"
    ].transform("sum")
    per_fold["family_abs_shap_fraction"] = np.divide(
        per_fold["family_mean_abs_shap"],
        totals,
        out=np.zeros(len(per_fold), dtype=float),
        where=totals.to_numpy() > 0,
    )
    summary_columns = [
        "experiment_id",
        "input_id",
        "window_sec",
        "overlap_sec",
        "target",
        "input_mode",
        "representation",
        "model",
        "family",
    ]
    return (
        per_fold.groupby(summary_columns, dropna=False, sort=False)
        .agg(
            mean_family_abs_shap_across_subjects=("family_mean_abs_shap", "mean"),
            std_family_abs_shap_across_subjects=("family_mean_abs_shap", "std"),
            median_family_abs_shap_across_subjects=(
                "family_mean_abs_shap",
                "median",
            ),
            mean_family_share_across_subjects=("family_abs_shap_fraction", "mean"),
            std_family_share_across_subjects=("family_abs_shap_fraction", "std"),
            feature_count=("family_feature_count", "max"),
            evaluated_subjects=("subject_id", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["experiment_id", "mean_family_abs_shap_across_subjects"],
            ascending=[True, False],
            kind="stable",
        )
    )


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "xgboost": xgboost.__version__,
    }


def write_tables(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    importance: list[dict[str, Any]],
    shap_importance: list[dict[str, Any]],
    shap_values: list[dict[str, Any]],
    progress: dict[str, Any]
) -> None:
    atomic_write_csv(pd.DataFrame(summaries), run_dir / "summary.csv")
    atomic_write_csv(pd.DataFrame(folds), run_dir / "fold_metrics.csv")
    atomic_write_csv(pd.DataFrame(predictions), run_dir / "oof_predictions.csv")
    if importance:
        importance_frame = pd.DataFrame(importance)
        atomic_write_csv(importance_frame, run_dir / "permutation_importance_folds.csv")
        atomic_write_csv(
            importance_summary(importance_frame),
            run_dir / "permutation_importance_summary.csv",
        )
    if shap_importance:
        shap_frame = pd.DataFrame(shap_importance)
        shap_summary = shap_importance_summary(shap_frame)
        atomic_write_csv(shap_frame, run_dir / "shap_importance_folds.csv")
        atomic_write_csv(shap_summary, run_dir / "shap_importance_summary.csv")
        atomic_write_csv(
            shap_summary.loc[
                shap_summary["rank_by_mean_abs_shap"]
                <= shap_summary["top_k"]
            ],
            run_dir / "shap_top_features.csv",
        )
        atomic_write_csv(
            shap_family_summary(shap_frame),
            run_dir / "shap_family_summary.csv",
        )
    if shap_values:
        atomic_write_csv(pd.DataFrame(shap_values), run_dir / "shap_values.csv")
    atomic_write_json(progress, run_dir / "progress.json")


def prepare_inputs(args: argparse.Namespace) -> tuple[list[InputDataset], list[dict[str, Any]], list[dict[str, Any]]]:
    input_paths = resolve_inputs(args)
    inputs = [load_input(path, args.include_neutral_condition) for path in input_paths]
    if len({item.input_id for item in inputs}) != len(inputs):
        raise SimulationError("Input CSV stems must be unique because they identify settings")

    participant_eligibility = apply_subject_completeness_policy(
        inputs, args.min_valid_trials_per_subject
    )
    input_eligibility = apply_common_trial_policy(inputs, args.trial_policy)
    return inputs, participant_eligibility, input_eligibility


def build_experiment_matrix(inputs: Sequence[InputDataset], args: argparse.Namespace) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(item, target, mode, representation, model)
        for item in inputs
        for target in args.targets
        for mode in args.input_modes
        for representation in args.representations
        for model in args.models
        if model not in GRAPH_MODEL_NAMES
        or representation in GRAPH_REPRESENTATIONS
    ]


def configuration_fingerprint(inputs: Sequence[InputDataset], args: argparse.Namespace) -> str:
    payload = {
        "inputs": [item.sha256 for item in inputs],
        "targets": args.targets,
        "modes": args.input_modes,
        "representations": args.representations,
        "models": args.models,
        "seed": args.seed,
        "inner_splits": args.inner_splits,
        "trial_policy": args.trial_policy,
        "min_valid_trials_per_subject": args.min_valid_trials_per_subject,
        "include_neutral_condition": args.include_neutral_condition,
        "bootstrap_repeats": args.bootstrap_repeats,
        "permutation_repeats": args.permutation_repeats,
        "statistical_permutation_repeats": args.statistical_permutation_repeats,
        "importance_models": args.importance_models,
        "shap_enabled": not args.no_shap,
        "shap_models": args.shap_models,
        "shap_background_size": args.shap_background_size,
        "shap_max_test_samples": args.shap_max_test_samples,
        "shap_top_k": args.shap_top_k,
        "save_shap_values": args.save_shap_values,
        "correlations_enabled": not args.no_correlations,
        "quick": args.quick,
        "max_outer_folds": args.max_outer_folds,
        "device": args.device,
        "graph_device": args.graph_device,
        "graph_batch_size": args.graph_batch_size,
        "graph_eval_batch_size": args.graph_eval_batch_size,
        "graph_max_epochs": args.graph_max_epochs,
        "graph_min_epochs": args.graph_min_epochs,
        "graph_patience": args.graph_patience,
        "graph_min_delta": args.graph_min_delta,
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:10]


def create_run_directory(args: argparse.Namespace, fingerprint: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = safe_token(args.run_name or f"{timestamp}_{fingerprint}")
    output_parent = args.output_dir.expanduser()
    if not output_parent.is_absolute():
        output_parent = Path.cwd() / output_parent
    run_dir = output_parent.resolve() / run_name
    if run_dir.exists():
        raise SimulationError(
            f"Run directory already exists: {run_dir}. Choose another --run-name; "
            "existing evidence will not be overwritten."
        )
    run_dir.mkdir(parents=True)
    return run_dir


def print_run_start(args: argparse.Namespace, inputs: Sequence[InputDataset], experiment_count: int, run_dir: Path) -> None:
    print("=" * 78)
    print("Emognition reproducible binary model simulation")
    print(f"Inputs:              {len(inputs)}")
    print(f"Experiments:         {experiment_count}")
    print(f"Targets:             {', '.join(args.targets)}")
    print(f"Input modes:         {', '.join(args.input_modes)}")
    print(f"Representations:     {', '.join(args.representations)}")
    print(f"Models:              {', '.join(args.models)}")
    print(
        "SHAP:                "
        + ("disabled" if args.no_shap else ", ".join(args.shap_models))
    )
    print(f"Minimum valid trials:{args.min_valid_trials_per_subject:>3}")
    print(f"Output:              {run_dir}")
    print(f"Deterministic seed:  {args.seed}")
    print("Nested LOSO can take many hours. Progress is printed for every held-out subject.")
    if args.quick or args.max_outer_folds:
        print("DIAGNOSTIC MODE: quick/partial outputs must not be reported as final results.")
    print("=" * 78, flush=True)


def manifest_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "targets": args.targets,
        "input_modes": args.input_modes,
        "representations": args.representations,
        "models": args.models,
        "seed": args.seed,
        "n_jobs": args.n_jobs,
        "inner_splits": args.inner_splits,
        "trial_policy": args.trial_policy,
        "min_valid_trials_per_subject": args.min_valid_trials_per_subject,
        "include_neutral_condition": args.include_neutral_condition,
        "bootstrap_repeats": args.bootstrap_repeats,
        "permutation_repeats": args.permutation_repeats,
        "importance_models": args.importance_models,
        "shap_enabled": not args.no_shap,
        "shap_models": args.shap_models,
        "shap_background_size": args.shap_background_size,
        "shap_max_test_samples": args.shap_max_test_samples,
        "shap_top_k": args.shap_top_k,
        "save_shap_values": args.save_shap_values,
        "correlations_enabled": not args.no_correlations,
        "quick": args.quick,
        "max_outer_folds": args.max_outer_folds,
        "device": args.device,
        "graph_device": args.graph_device,
        "graph_batch_size": args.graph_batch_size,
        "graph_eval_batch_size": args.graph_eval_batch_size,
        "graph_max_epochs": args.graph_max_epochs,
        "graph_min_epochs": args.graph_min_epochs,
        "graph_patience": args.graph_patience,
        "graph_min_delta": args.graph_min_delta,
    }


def analysis_contract(args: argparse.Namespace) -> dict[str, str]:
    conditions = (
        "BASELINE excluded; NEUTRAL retained only for explicit sensitivity"
        if args.include_neutral_condition
        else "BASELINE and NEUTRAL excluded"
    )
    return {
        "conditions": conditions,
        "subject_completeness": (
            "one cohort across all settings; exclude subjects with fewer than "
            f"{args.min_valid_trials_per_subject} valid non-BASELINE/non-NEUTRAL "
            "trials in any input; criterion is independent of labels and model results"
        ),
        "phase": "STIMULUS only (enforced by upstream feature extractor)",
        "binary_labels": "target-specific SAM 5 excluded; 1--4=low(0), 6--9=high(1)",
        "outer_evaluation": "LeaveOneGroupOut by subject",
        "inner_selection": "StratifiedGroupKFold when valid, else GroupKFold; subject groups",
        "fold_preprocessing": (
            "training-fold median imputation with empty features retained; "
            "training-fold VarianceThreshold(0); StandardScaler for SVM/KNN/MLP; "
            "GCN/GAT use training-fold median imputation and scaling without "
            "dimension-changing variance removal"
        ),
        "imbalance": (
            "balanced class weights for SVM/RF; each fitted XGBoost training "
            "partition sets scale_pos_weight from its own negative/positive ratio; "
            "each MLP fit uses balanced sample weights; each GCN/GAT fit uses "
            "training-partition class-weighted cross entropy; "
            "no synthetic oversampling"
        ),
        "aggregation": "per-trial population mean and population standard deviation",
        "graph_definition": (
            "GCN/GAT use three fully connected directed modality nodes "
            "(BVP, EDA, TEMP), no explicit self-edges, aligned NVG/HVG summary "
            "statistics plus a three-element modality identity as node attributes"
        ),
        "graph_training": (
            f"residual graph layers with LayerNorm; validation-loss checkpointing "
            f"starts at epoch {args.graph_min_epochs}; patience="
            f"{args.graph_patience}; maximum epochs={args.graph_max_epochs}"
        ),
        "primary_metric": (
            "pooled out-of-fold binary balanced accuracy; fold-level BA is undefined "
            "when the held-out subject contains only one target class"
        ),
        "confidence_intervals": "subject-cluster percentile bootstrap",
        "statistical_tests": (
            "paired subject-cluster permutation of pooled OOF accuracy and balanced "
            "accuracy with Holm correction"
        ),
        "importance": "permutation importance on each untouched outer test subject",
        "shap": (
            "SHAP on each untouched outer test subject using only an outer-training "
            "background; subject-balanced magnitude, direction, family contribution, "
            "rank, and top-k stability summaries"
        ),
        "correlations": "descriptive Spearman correlations only; never used for selection",
    }


def input_manifest_rows(inputs: Sequence[InputDataset]) -> list[dict[str, Any]]:
    return [
        {
            "input_id": item.input_id,
            "path_hint": item.path.name,
            "sha256": item.sha256,
            "manifest_path_hint": item.manifest_path.name if item.manifest_path else None,
            "manifest_sha256": item.manifest_sha256,
            "window_sec": item.window_sec,
            "overlap_sec": item.overlap_sec,
            "rows_after_policy": len(item.table),
            "trials_after_policy": item.table["trial_id"].nunique(),
            "subjects_after_policy": item.table["subject_id"].nunique(),
        }
        for item in inputs
    ]


def build_run_manifest(args: argparse.Namespace, inputs: Sequence[InputDataset], participant_eligibility: list[dict[str, Any]], fingerprint: str, started_utc: str) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
        "status": "running",
        "started_utc": started_utc,
        "command_argv": [str(script_path), *sys.argv[1:]],
        "script_sha256": sha256_file(script_path),
        "configuration_hash": fingerprint,
        "parameters": manifest_parameters(args),
        "analysis_contract": analysis_contract(args),
        "feature_schema": {
            "expected_counts": EXPECTED_FEATURE_COUNTS,
            "representations": REPRESENTATIONS,
        },
        "inputs": input_manifest_rows(inputs),
        "dependencies": dependency_versions(),
        "subject_eligibility": participant_eligibility,
    }


def write_initial_run_files(run_dir: Path, manifest: dict[str, Any], participant_eligibility: list[dict[str, Any]], input_eligibility: list[dict[str, Any]]) -> None:
    atomic_write_json(manifest, run_dir / "run_manifest.json")
    atomic_write_csv(
        pd.DataFrame(participant_eligibility),
        run_dir / "participant_eligibility.csv",
    )
    atomic_write_csv(pd.DataFrame(input_eligibility), run_dir / "input_eligibility.csv")


def write_class_distribution(run_dir: Path, inputs: Sequence[InputDataset], targets: Sequence[str]) -> None:
    rows = [
        row
        for item in inputs
        for target in targets
        for row in distribution_rows(item, target)
    ]
    atomic_write_csv(pd.DataFrame(rows), run_dir / "class_distribution.csv")


def write_correlation_tables(run_dir: Path, inputs: Sequence[InputDataset]) -> None:
    correlations = pd.concat(
        [descriptive_correlations(item) for item in inputs], ignore_index=True
    )
    atomic_write_csv(correlations, run_dir / "feature_correlations_spearman.csv")
    high_cross_family = correlations.loc[
        (~correlations["same_family"])
        & (correlations["abs_spearman_rho"] >= 0.8)
    ].sort_values("abs_spearman_rho", ascending=False, kind="stable")
    atomic_write_csv(
        high_cross_family,
        run_dir / "high_cross_family_correlations.csv",
    )


def write_descriptive_tables(run_dir: Path, inputs: Sequence[InputDataset], args: argparse.Namespace) -> None:
    write_class_distribution(run_dir, inputs, args.targets)
    if not args.no_correlations:
        write_correlation_tables(run_dir, inputs)


def execute_experiment_matrix(run_dir: Path, experiment_matrix: Sequence[ExperimentSpec], args: argparse.Namespace, started: float) -> ExperimentResults:
    results = ExperimentResults([], [], [], [], [], [])
    total_experiments = len(experiment_matrix)
    for index, experiment in enumerate(experiment_matrix, start=1):
        (
            summary,
            folds,
            predictions,
            importance,
            shap_importance,
            shap_values,
        ) = run_experiment(
            experiment.dataset,
            experiment.target,
            experiment.input_mode,
            experiment.representation,
            experiment.model_name,
            args,
            index,
            total_experiments,
        )
        results.summaries.append(summary)
        results.folds.extend(folds)
        results.predictions.extend(predictions)
        results.importance.extend(importance)
        results.shap_importance.extend(shap_importance)
        results.shap_values.extend(shap_values)
        write_tables(
            run_dir,
            results.summaries,
            results.folds,
            results.predictions,
            results.importance,
            results.shap_importance,
            results.shap_values,
            {
                "status": "running",
                "completed_experiments": index,
                "total_experiments": total_experiments,
                "elapsed_seconds": time.perf_counter() - started,
                "last_experiment_id": summary["experiment_id"],
            },
        )
    return results


def complete_run(run_dir: Path, manifest: dict[str, Any], results: ExperimentResults, args: argparse.Namespace, started: float, experiment_count: int) -> float:
    tests = paired_tests(
        pd.DataFrame(results.predictions),
        repeats=args.statistical_permutation_repeats,
        seed=args.seed,
    )
    atomic_write_csv(tests, run_dir / "paired_statistical_tests.csv")

    finished_utc = datetime.now(timezone.utc).isoformat()
    total_seconds = time.perf_counter() - started
    manifest.update(
        {
            "status": "complete",
            "finished_utc": finished_utc,
            "total_run_seconds": total_seconds,
            "experiment_count": experiment_count,
            "complete_loso_for_all_experiments": args.max_outer_folds == 0,
            "reportable_final_run": bool(not args.quick and args.max_outer_folds == 0 and args.bootstrap_repeats > 0)
        }
    )
    atomic_write_json(manifest, run_dir / "run_manifest.json")
    atomic_write_json(
        {
            "status": "complete",
            "completed_experiments": experiment_count,
            "total_experiments": experiment_count,
            "elapsed_seconds": total_seconds,
            "finished_utc": finished_utc,
        },
        run_dir / "progress.json",
    )
    return total_seconds


def print_run_complete(run_dir: Path, experiment_count: int, total_seconds: float) -> None:
    print("=" * 78)
    print(f"Complete: {experiment_count} experiments in {total_seconds / 60:.1f} min")
    print(f"Results:  {run_dir}")
    print("Primary table: summary.csv; predictions: oof_predictions.csv")
    print("=" * 78, flush=True)


def run(args: argparse.Namespace) -> int:
    validate_args(args)
    inputs, participant_eligibility, input_eligibility = prepare_inputs(args)
    experiment_matrix = build_experiment_matrix(inputs, args)
    fingerprint = configuration_fingerprint(inputs, args)
    run_dir = create_run_directory(args, fingerprint)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    print_run_start(args, inputs, len(experiment_matrix), run_dir)
    manifest = build_run_manifest(
        args,
        inputs,
        participant_eligibility,
        fingerprint,
        started_utc,
    )
    write_initial_run_files(
        run_dir,
        manifest,
        participant_eligibility,
        input_eligibility,
    )
    write_descriptive_tables(run_dir, inputs, args)
    results = execute_experiment_matrix(
        run_dir,
        experiment_matrix,
        args,
        started,
    )
    total_seconds = complete_run(
        run_dir,
        manifest,
        results,
        args,
        started,
        len(experiment_matrix),
    )
    print_run_complete(run_dir, len(experiment_matrix), total_seconds)
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (SimulationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
