from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ..config_types import AppConfig
from ..data.transforms import build_model_table
from ..domain import CandidateSpec, Dataset, NestedTargetResult
from ..errors import ExperimentError
from ..explainability import permutation_rows, shap_rows
from ..models import build_model
from ..models.registry import GRAPH_MODELS
from ..tasks import TaskDefinition
from ..utils import safe_token, stable_seed
from .aggregation import (
    aggregate_predictions_to_trial,
    subject_macro_primary_score,
)
from .metrics import bootstrap_intervals, compute_metrics, prediction_scores
from .runtime import PhaseProgressReporter, ProgressReporter, TargetCheckpointStore
from .splitters import shared_inner_subject_splits, subject_sort_key


@dataclass
class PreparedCandidate:
    spec: CandidateSpec
    frame: pd.DataFrame
    features: list[str]


@dataclass
class CandidateChoice:
    candidate: PreparedCandidate
    parameters: dict[str, Any]
    score: float
    utility: float
    parameter_index: int
    inner_fit_seconds: float = 0.0


def expand_candidate_specs(
    datasets: list[Dataset], config: AppConfig
) -> list[CandidateSpec]:
    by_id = {dataset.input_id: dataset for dataset in datasets}
    specs: list[CandidateSpec] = []
    order = 0
    for template in config.selection.candidates:
        if template.input_id == "*":
            selected = datasets
        else:
            if template.input_id not in by_id:
                raise ExperimentError(
                    f"Candidate {template.name!r} references unknown input_id "
                    f"{template.input_id!r}; available: {sorted(by_id)}"
                )
            selected = [by_id[template.input_id]]
        if (
            template.model_name in GRAPH_MODELS
            and template.representation not in config.models.graph_representations
        ):
            raise ExperimentError(
                f"Candidate {template.name!r}: {template.model_name} does not support "
                f"representation {template.representation!r}"
            )
        for dataset in selected:
            candidate_id = safe_token(f"{template.name}__{dataset.input_id}")
            specs.append(
                CandidateSpec(
                    candidate_id=candidate_id,
                    order=order,
                    dataset=dataset,
                    input_mode=template.input_mode,
                    representation=template.representation,
                    model_name=template.model_name,
                )
            )
            order += 1
    identifiers = [spec.candidate_id for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier for identifier, count in Counter(identifiers).items() if count > 1
        )
        raise ExperimentError(f"Expanded candidate IDs are not unique: {duplicates}")
    if not specs:
        raise ExperimentError("Candidate expansion produced no pipelines")
    return specs


def _prepare_candidates(
    specs: list[CandidateSpec],
    target_name: str,
    task: TaskDefinition,
    config: AppConfig,
) -> tuple[list[PreparedCandidate], set[str]]:
    table_cache: dict[tuple[str, str, str], tuple[pd.DataFrame, list[str]]] = {}
    prepared: list[PreparedCandidate] = []
    for spec in specs:
        key = (spec.dataset.input_id, spec.input_mode, spec.representation)
        if key not in table_cache:
            table_cache[key] = build_model_table(
                spec.dataset,
                target_name,
                spec.input_mode,
                spec.representation,
                task,
                config,
            )
        frame, features = table_cache[key]
        prepared.append(PreparedCandidate(spec, frame.copy(), list(features)))

    common_trials = set.intersection(
        *(set(candidate.frame["trial_id"].astype(str)) for candidate in prepared)
    )
    if not common_trials:
        raise ExperimentError(
            f"No common target-eligible trials exist across candidates for {target_name}"
        )
    for candidate in prepared:
        candidate.frame = candidate.frame.loc[
            candidate.frame["trial_id"].astype(str).isin(common_trials)
        ].copy()
        candidate.frame[config.schema.subject_column] = candidate.frame[
            config.schema.subject_column
        ].astype(str)

    reference = prepared[0].frame.drop_duplicates("trial_id").set_index("trial_id")["y"]
    for candidate in prepared[1:]:
        labels = candidate.frame.drop_duplicates("trial_id").set_index("trial_id")["y"]
        shared = sorted(common_trials)
        if not np.array_equal(
            reference.loc[shared].to_numpy(), labels.loc[shared].to_numpy()
        ):
            raise ExperimentError(
                f"Encoded labels differ across candidates for target {target_name}"
            )
    return prepared, common_trials


def _fit_candidate(
    candidate: PreparedCandidate,
    task: TaskDefinition,
    config: AppConfig,
    train_mask: np.ndarray,
    parameters: dict[str, Any],
    seed: int,
) -> tuple[Any, Any]:
    frame = candidate.frame
    X = frame[candidate.features].astype(float)
    y = frame["y"].to_numpy(dtype=int if task.is_classification else float)
    groups = frame[config.schema.subject_column].astype(str).to_numpy()
    bundle = build_model(
        candidate.spec.model_name,
        task,
        candidate.features,
        candidate.spec.representation,
        config,
        seed,
    )
    if task.is_classification and np.unique(y[train_mask]).size != task.class_count:
        raise ExperimentError(
            f"Candidate {candidate.spec.candidate_id} training partition is missing "
            "a configured class"
        )
    estimator = bundle.estimator.set_params(**parameters)
    fit_parameters = {"clf__groups": groups[train_mask]} if bundle.is_graph else {}
    estimator.fit(X.loc[train_mask], y[train_mask], **fit_parameters)
    return estimator, bundle


def _predict_candidate(
    estimator: Any,
    candidate: PreparedCandidate,
    task: TaskDefinition,
    config: AppConfig,
    mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    frame = candidate.frame.loc[mask]
    X = frame[candidate.features].astype(float)
    y = frame["y"].to_numpy(dtype=int if task.is_classification else float)
    predicted = np.asarray(estimator.predict(X))
    scores = prediction_scores(estimator, X, task)
    trials = aggregate_predictions_to_trial(
        frame,
        y,
        predicted,
        scores,
        task,
        config.schema.subject_column,
    )
    samples = pd.DataFrame(
        {
            "subject_id": frame[config.schema.subject_column].astype(str).to_numpy(),
            "trial_id": frame["trial_id"].astype(str).to_numpy(),
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "target_raw": frame["target_raw"].to_numpy(),
            "y_true": y,
            "y_pred": predicted,
        }
    )
    score_array = np.asarray(scores)
    if not task.is_classification or score_array.ndim == 1:
        samples["y_score"] = score_array.reshape(-1)
    elif task.class_count == 2:
        positive = task.positive_index()
        positive = 1 if positive is None else positive
        samples["y_score"] = score_array[:, positive]
    else:
        samples["y_scores_json"] = [
            json.dumps(row.astype(float).tolist()) for row in score_array
        ]
    return trials, samples, y, score_array


def _trial_score_matrix(
    predictions: pd.DataFrame, task: TaskDefinition
) -> np.ndarray | None:
    if not task.is_classification:
        return predictions["y_score"].to_numpy(float)
    if task.class_count == 2 and "y_score" in predictions:
        positive = task.positive_index()
        positive = 1 if positive is None else positive
        values = predictions["y_score"].to_numpy(float)
        if positive == 1:
            return np.column_stack([1.0 - values, values])
        return np.column_stack([values, 1.0 - values])
    if "y_scores_json" in predictions:
        return np.stack(predictions["y_scores_json"].map(json.loads))
    return None


def _split_fingerprint(folds: list[Any]) -> str:
    payload = [
        {
            "train": fold.train_subjects,
            "validation": fold.validation_subjects,
        }
        for fold in folds
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _candidate_parameter_grid(
    candidate: PreparedCandidate,
    task: TaskDefinition,
    config: AppConfig,
    seed: int,
) -> list[dict[str, Any]]:
    bundle = build_model(
        candidate.spec.model_name,
        task,
        candidate.features,
        candidate.spec.representation,
        config,
        seed,
    )
    return list(ParameterGrid(bundle.parameter_grid))


def _tune_inner_candidates(
    prepared: list[PreparedCandidate],
    reference: pd.DataFrame,
    outer_subject: str,
    outer_fold: int,
    target_name: str,
    task: TaskDefinition,
    config: AppConfig,
    checkpoint: TargetCheckpointStore,
    progress: ProgressReporter,
    outer_total: int,
) -> tuple[
    list[CandidateChoice],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    int,
]:
    subject_column = config.schema.subject_column
    outer_train_reference = reference.loc[
        reference[subject_column].astype(str) != outer_subject
    ].copy()
    split_seed = stable_seed(config.evaluation.seed, target_name, outer_fold, "inner_split")
    folds, splitter_name, split_count = shared_inner_subject_splits(
        outer_train_reference,
        subject_column,
        config.evaluation.inner_splits,
        split_seed,
        task,
    )
    split_hash = _split_fingerprint(folds)
    split_rows = [
        {
            "target": target_name,
            "task_kind": task.config.kind,
            "outer_fold": outer_fold,
            "outer_subject_id": outer_subject,
            "inner_fold": inner_fold,
            "inner_splitter": splitter_name,
            "inner_splits": split_count,
            "inner_split_fingerprint": split_hash,
            "train_subjects_json": json.dumps(fold.train_subjects),
            "validation_subjects_json": json.dumps(fold.validation_subjects),
        }
        for inner_fold, fold in enumerate(folds, start=1)
    ]
    rows: list[dict[str, Any]] = []
    choices: list[CandidateChoice] = []

    for candidate in prepared:
        frame_subjects = candidate.frame[subject_column].astype(str)
        parameter_grid = _candidate_parameter_grid(
            candidate,
            task,
            config,
            stable_seed(split_seed, candidate.spec.candidate_id, "grid"),
        )
        saved = checkpoint.load_inner(
            outer_fold, outer_subject, candidate.spec
        )
        if saved is not None:
            if saved.get("inner_split_fingerprint") != split_hash:
                raise ExperimentError(
                    "Checkpoint inner split does not match the current split plan for "
                    f"{candidate.spec.candidate_id}"
                )
            saved_rows = list(saved.get("rows", []))
            rows.extend(saved_rows)
            choices.append(
                CandidateChoice(
                    candidate=candidate,
                    parameters=dict(saved["parameters"]),
                    score=float(saved["score"]),
                    utility=float(saved["utility"]),
                    parameter_index=int(saved["parameter_index"]),
                    inner_fit_seconds=float(saved.get("inner_fit_seconds", 0.0)),
                )
            )
            progress.advance(
                len(parameter_grid) * len(folds),
                phase="resume inner",
                outer_fold=outer_fold,
                outer_total=outer_total,
                candidate=candidate.spec.candidate_id,
                force=True,
            )
            continue

        candidate_choices: list[CandidateChoice] = []
        candidate_rows: list[dict[str, Any]] = []
        candidate_fit_seconds = 0.0
        for parameter_index, parameters in enumerate(parameter_grid):
            validation_predictions: list[pd.DataFrame] = []
            fold_scores: list[float] = []
            fit_seconds = 0.0
            for inner_fold, fold in enumerate(folds, start=1):
                progress.advance(
                    0,
                    phase=(
                        f"inner p{parameter_index + 1}/{len(parameter_grid)} "
                        f"fold {inner_fold}/{len(folds)}"
                    ),
                    outer_fold=outer_fold,
                    outer_total=outer_total,
                    candidate=candidate.spec.candidate_id,
                    force=parameter_index == 0 and inner_fold == 1,
                )
                train_mask = frame_subjects.isin(fold.train_subjects).to_numpy()
                validation_mask = frame_subjects.isin(
                    fold.validation_subjects
                ).to_numpy()
                if not train_mask.any() or not validation_mask.any():
                    raise ExperimentError(
                        f"Candidate {candidate.spec.candidate_id} has an empty inner partition"
                    )
                seed = stable_seed(
                    config.evaluation.seed,
                    target_name,
                    outer_fold,
                    candidate.spec.candidate_id,
                    parameter_index,
                    inner_fold,
                )
                started = time.perf_counter()
                estimator, _ = _fit_candidate(
                    candidate, task, config, train_mask, parameters, seed
                )
                elapsed_fit = time.perf_counter() - started
                fit_seconds += elapsed_fit
                candidate_fit_seconds += elapsed_fit
                trial_predictions, _, _, _ = _predict_candidate(
                    estimator, candidate, task, config, validation_mask
                )
                progress.advance(
                    1,
                    phase=(
                        f"inner p{parameter_index + 1}/{len(parameter_grid)} "
                        f"fold {inner_fold}/{len(folds)}"
                    ),
                    outer_fold=outer_fold,
                    outer_total=outer_total,
                    candidate=candidate.spec.candidate_id,
                )
                trial_predictions["inner_fold"] = inner_fold
                validation_predictions.append(trial_predictions)
                try:
                    fold_score, _, _ = subject_macro_primary_score(
                        trial_predictions, task
                    )
                except ExperimentError:
                    fold_score = math.nan
                fold_scores.append(fold_score)

            combined = pd.concat(validation_predictions, ignore_index=True)
            score, evaluable, excluded = subject_macro_primary_score(combined, task)
            utility = score if task.is_classification else -score
            choice = CandidateChoice(
                candidate,
                dict(parameters),
                score,
                utility,
                parameter_index,
                candidate_fit_seconds,
            )
            candidate_choices.append(choice)
            if config.output.save_inner_search_details:
                candidate_rows.append(
                    {
                        "target": target_name,
                        "task_kind": task.config.kind,
                        "outer_fold": outer_fold,
                        "outer_subject_id": outer_subject,
                        "candidate_id": candidate.spec.candidate_id,
                        "candidate_order": candidate.spec.order,
                        "input_id": candidate.spec.dataset.input_id,
                        "input_mode": candidate.spec.input_mode,
                        "representation": candidate.spec.representation,
                        "model": candidate.spec.model_name,
                        "parameter_index": parameter_index,
                        "parameters_json": json.dumps(
                            parameters, sort_keys=True, default=str
                        ),
                        "selection_metric": (
                            config.selection.classification_metric
                            if task.is_classification
                            else config.selection.regression_metric
                        ),
                        "selection_score": score,
                        "selection_utility": utility,
                        "evaluable_subjects": evaluable,
                        "primary_excluded_subjects": excluded,
                        "inner_fold_scores_json": json.dumps(fold_scores),
                        "inner_splitter": splitter_name,
                        "inner_splits": split_count,
                        "inner_split_fingerprint": split_hash,
                        "fit_seconds": fit_seconds,
                    }
                )
        best_for_candidate = max(
            candidate_choices,
            key=lambda choice: (choice.utility, -choice.parameter_index),
        )
        best_for_candidate.inner_fit_seconds = candidate_fit_seconds
        choices.append(best_for_candidate)
        for row in candidate_rows:
            selected = row["parameter_index"] == best_for_candidate.parameter_index
            row["best_parameter_for_candidate"] = bool(selected)
            row["selected_parameter_for_candidate"] = bool(selected)
        rows.extend(candidate_rows)
        checkpoint.save_inner(
            outer_fold,
            outer_subject,
            candidate.spec,
            {
                "inner_split_fingerprint": split_hash,
                "parameters": best_for_candidate.parameters,
                "score": best_for_candidate.score,
                "utility": best_for_candidate.utility,
                "parameter_index": best_for_candidate.parameter_index,
                "inner_fit_seconds": candidate_fit_seconds,
                "rows": candidate_rows,
            },
        )
        checkpoint.write_progress(
            progress,
            phase="inner_candidate_complete",
            outer_fold=outer_fold,
            outer_subject_id=outer_subject,
            candidate_id=candidate.spec.candidate_id,
        )

    best_parameter_indices = {
        choice.candidate.spec.candidate_id: choice.parameter_index
        for choice in choices
    }
    for row in rows:
        row["best_parameter_for_candidate"] = bool(
            row["parameter_index"]
            == best_parameter_indices[row["candidate_id"]]
        )
        row["selected_parameter_for_candidate"] = bool(
            row["parameter_index"]
            == best_parameter_indices[row["candidate_id"]]
        )
    return choices, rows, split_rows, splitter_name, split_count


def run_nested_target(
    datasets: list[Dataset],
    candidate_specs: list[CandidateSpec],
    target_name: str,
    task: TaskDefinition,
    config: AppConfig,
    run_dir: Path | None = None,
    target_index: int = 1,
    target_count: int = 1,
) -> NestedTargetResult:
    started = time.perf_counter()
    prepared, common_trials = _prepare_candidates(
        candidate_specs, target_name, task, config
    )
    subject_column = config.schema.subject_column
    reference = prepared[0].frame.drop_duplicates("trial_id")[
        [subject_column, "trial_id", "y"]
    ].copy()
    reference[subject_column] = reference[subject_column].astype(str)
    subjects = sorted(reference[subject_column].unique(), key=subject_sort_key)
    outer_subjects = subjects[:]
    if config.evaluation.max_outer_folds:
        outer_subjects = outer_subjects[: config.evaluation.max_outer_folds]
    grid_units = sum(
        len(
            _candidate_parameter_grid(
                candidate,
                task,
                config,
                stable_seed(
                    config.evaluation.seed,
                    target_name,
                    candidate.spec.candidate_id,
                    "progress_grid",
                ),
            )
        )
        for candidate in prepared
    )
    total_fit_units = 0
    for progress_fold, progress_subject in enumerate(outer_subjects, start=1):
        progress_reference = reference.loc[
            reference[subject_column].astype(str) != progress_subject
        ].copy()
        progress_seed = stable_seed(
            config.evaluation.seed,
            target_name,
            progress_fold,
            "inner_split",
        )
        progress_folds, _, _ = shared_inner_subject_splits(
            progress_reference,
            subject_column,
            config.evaluation.inner_splits,
            progress_seed,
            task,
        )
        total_fit_units += grid_units * len(progress_folds) + len(prepared)

    progress = ProgressReporter(
        total_fit_units,
        target_name,
        target_index,
        target_count,
        run_dir,
        enabled=config.evaluation.show_progress,
    )
    checkpoint = TargetCheckpointStore(
        run_dir=run_dir or Path("."),
        target_name=target_name,
        task_kind=task.config.kind,
        enabled=bool(run_dir is not None and config.output.save_checkpoints),
    )
    progress.advance(
        0,
        phase="starting",
        outer_fold=0,
        outer_total=len(outer_subjects),
        candidate="-",
        force=True,
    )
    progress.log(
        f"target={target_name} started | candidates={len(candidate_specs)} | "
        f"outer_subjects={len(outer_subjects)}/{len(subjects)} | "
        f"common_trials={len(common_trials)}",
        console=True,
    )

    inner_rows: list[dict[str, Any]] = []
    split_assignment_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    permutation_records: list[dict[str, Any]] = []
    shap_records: list[dict[str, Any]] = []
    shap_value_records: list[dict[str, Any]] = []

    for outer_fold, outer_subject in enumerate(outer_subjects, start=1):
        progress.log(
            f"target={target_name} | outer={outer_fold}/{len(outer_subjects)} | "
            f"subject={outer_subject} | tuning_started",
        )
        choices, candidate_rows, split_rows, splitter_name, split_count = (
            _tune_inner_candidates(
                prepared,
                reference,
                outer_subject,
                outer_fold,
                target_name,
                task,
                config,
                checkpoint,
                progress,
                len(outer_subjects),
            )
        )
        inner_rows.extend(candidate_rows)
        if config.output.save_split_assignments:
            split_assignment_rows.extend(split_rows)

        for choice in choices:
            candidate_started = time.perf_counter()
            candidate = choice.candidate
            saved_outer = checkpoint.load_outer(
                outer_fold, outer_subject, candidate.spec
            )
            if saved_outer is not None:
                fold_rows.extend(saved_outer.get("folds", []))
                prediction_rows.extend(saved_outer.get("predictions", []))
                window_rows.extend(saved_outer.get("window_predictions", []))
                permutation_records.extend(
                    saved_outer.get("permutation_importance", [])
                )
                shap_records.extend(saved_outer.get("shap_importance", []))
                shap_value_records.extend(saved_outer.get("shap_values", []))
                progress.advance(
                    1,
                    phase="resume outer",
                    outer_fold=outer_fold,
                    outer_total=len(outer_subjects),
                    candidate=candidate.spec.candidate_id,
                    force=True,
                )
                continue

            progress.advance(
                0,
                phase="outer refit",
                outer_fold=outer_fold,
                outer_total=len(outer_subjects),
                candidate=candidate.spec.candidate_id,
                force=True,
            )
            frame_subjects = candidate.frame[subject_column].astype(str)
            train_mask = (frame_subjects != outer_subject).to_numpy()
            test_mask = (frame_subjects == outer_subject).to_numpy()
            if not test_mask.any():
                raise ExperimentError(
                    f"Candidate {candidate.spec.candidate_id} has no samples for "
                    f"outer subject {outer_subject}"
                )
            experiment_id = safe_token(
                f"nested__{config.schema.dataset_id}__{target_name}__"
                f"{task.config.kind}__{candidate.spec.candidate_id}"
            )
            fold_seed = stable_seed(
                config.evaluation.seed,
                target_name,
                outer_fold,
                candidate.spec.candidate_id,
                "outer_refit",
            )
            fit_started = time.perf_counter()
            estimator, bundle = _fit_candidate(
                candidate,
                task,
                config,
                train_mask,
                choice.parameters,
                fold_seed,
            )
            fit_seconds = time.perf_counter() - fit_started
            trials, samples, y_test, _ = _predict_candidate(
                estimator, candidate, task, config, test_mask
            )
            metadata = {
                "experiment_id": experiment_id,
                "dataset_id": config.schema.dataset_id,
                "target": target_name,
                "task_kind": task.config.kind,
                "outer_fold": outer_fold,
                "subject_id": outer_subject,
                "candidate_id": candidate.spec.candidate_id,
                "candidate_order": candidate.spec.order,
                "input_id": candidate.spec.dataset.input_id,
                "input_mode": candidate.spec.input_mode,
                "representation": candidate.spec.representation,
                "model": candidate.spec.model_name,
            }
            for key, value in metadata.items():
                trials[key] = value
                samples[key] = value
            candidate_predictions = trials.to_dict("records")
            candidate_windows: list[dict[str, Any]] = []
            if config.output.save_window_predictions:
                candidate_windows = samples.to_dict("records")

            trial_scores = _trial_score_matrix(trials, task)
            metrics = compute_metrics(
                trials["y_true"], trials["y_pred"], trial_scores, task
            )
            subject_score, evaluable, excluded = subject_macro_primary_score(
                trials, task
            )
            parameters_json = json.dumps(
                choice.parameters, sort_keys=True, default=str
            )
            candidate_folds = [
                {
                    **metadata,
                    "inner_splitter": splitter_name,
                    "inner_splits": split_count,
                    "inner_split_fingerprint": split_rows[0][
                        "inner_split_fingerprint"
                    ],
                    "best_parameters_json": parameters_json,
                    "best_inner_score": choice.score,
                    "inner_fit_seconds": choice.inner_fit_seconds,
                    "selection_metric": (
                        config.selection.classification_metric
                        if task.is_classification
                        else config.selection.regression_metric
                    ),
                    "outer_train_subjects": len(subjects) - 1,
                    "outer_test_trials": len(trials),
                    "outer_test_source_samples": len(samples),
                    "subject_primary_score": subject_score,
                    "subject_primary_evaluable": evaluable,
                    "subject_primary_excluded": excluded,
                    "fit_seconds": fit_seconds,
                    "fold_seconds": time.perf_counter() - candidate_started,
                    **metrics,
                }
            ]
            analysis_metadata = {**metadata, "fold": outer_fold}
            candidate_permutation: list[dict[str, Any]] = []
            candidate_shap: list[dict[str, Any]] = []
            candidate_shap_values: list[dict[str, Any]] = []
            importance_config = config.analyses.permutation_importance
            class_evaluable = (
                not task.is_classification
                or np.unique(y_test).size == task.class_count
            )
            if (
                importance_config.enabled
                and importance_config.repeats > 0
                and candidate.spec.model_name in importance_config.models
                and class_evaluable
            ):
                candidate_permutation.extend(
                    permutation_rows(
                        estimator,
                        candidate.frame.loc[test_mask, candidate.features].astype(float),
                        y_test,
                        candidate.features,
                        task,
                        importance_config.repeats,
                        fold_seed,
                        config.evaluation.n_jobs,
                        analysis_metadata,
                    )
                )
            shap_config = config.analyses.shap
            if (
                shap_config.enabled
                and candidate.spec.model_name in shap_config.models
                and bundle.supports_shap
            ):
                fold_shap, fold_values = shap_rows(
                    estimator,
                    candidate.frame.loc[train_mask, candidate.features].astype(float),
                    candidate.frame.loc[test_mask, candidate.features].astype(float),
                    candidate.features,
                    task,
                    shap_config.background_size,
                    shap_config.max_test_samples,
                    fold_seed,
                    shap_config.save_local_values,
                    analysis_metadata,
                )
                candidate_shap.extend(fold_shap)
                candidate_shap_values.extend(fold_values)

            outer_payload = {
                "folds": candidate_folds,
                "predictions": candidate_predictions,
                "window_predictions": candidate_windows,
                "permutation_importance": candidate_permutation,
                "shap_importance": candidate_shap,
                "shap_values": candidate_shap_values,
            }
            checkpoint.save_outer(
                outer_fold,
                outer_subject,
                candidate.spec,
                outer_payload,
            )
            fold_rows.extend(candidate_folds)
            prediction_rows.extend(candidate_predictions)
            window_rows.extend(candidate_windows)
            permutation_records.extend(candidate_permutation)
            shap_records.extend(candidate_shap)
            shap_value_records.extend(candidate_shap_values)
            progress.advance(
                1,
                phase="outer complete",
                outer_fold=outer_fold,
                outer_total=len(outer_subjects),
                candidate=candidate.spec.candidate_id,
                force=True,
            )
            checkpoint.write_progress(
                progress,
                phase="outer_candidate_complete",
                outer_fold=outer_fold,
                outer_subject_id=outer_subject,
                candidate_id=candidate.spec.candidate_id,
            )

            progress.log(
                f"target={target_name} | outer={outer_fold}/{len(outer_subjects)} | "
                f"subject={outer_subject} | candidate={candidate.spec.candidate_id} | "
                f"model={candidate.spec.model_name} | "
                f"representation={candidate.spec.representation} | "
                f"mode={candidate.spec.input_mode} | outer_complete | "
                f"primary_score={subject_score:.6g}",
            )

        progress.log(
            f"target={target_name} | outer={outer_fold}/{len(outer_subjects)} | "
            f"subject={outer_subject} | complete",
            console=True,
        )

    progress.close()

    predictions = pd.DataFrame(prediction_rows)
    tuned = pd.DataFrame(fold_rows)
    summaries: list[dict[str, Any]] = []
    bootstrap_repeats = config.evaluation.bootstrap_repeats
    bootstrap_progress = (
        PhaseProgressReporter(
            total=len(prepared) * bootstrap_repeats,
            phase_name=f"Bootstrap {target_index}/{target_count}: {target_name}",
            run_dir=run_dir,
            enabled=config.evaluation.show_progress,
            unit="repeat",
        )
        if bootstrap_repeats > 0
        else None
    )
    if bootstrap_progress is not None:
        bootstrap_progress.start()
    for candidate in prepared:
        candidate_predictions = predictions.loc[
            predictions["candidate_id"] == candidate.spec.candidate_id
        ].copy()
        if candidate_predictions.empty:
            raise ExperimentError(
                f"Candidate {candidate.spec.candidate_id} produced no outer predictions"
            )
        experiment_id = str(candidate_predictions["experiment_id"].iloc[0])
        pooled_scores = _trial_score_matrix(candidate_predictions, task)
        pooled = compute_metrics(
            candidate_predictions["y_true"],
            candidate_predictions["y_pred"],
            pooled_scores,
            task,
        )
        subject_primary, evaluable_subjects, excluded_subjects = (
            subject_macro_primary_score(candidate_predictions, task)
        )
        try:
            intervals = bootstrap_intervals(
                candidate_predictions,
                bootstrap_repeats,
                stable_seed(config.evaluation.seed, experiment_id, "bootstrap"),
                task,
                progress_callback=(
                    (
                        lambda completed, total, candidate_id=candidate.spec.candidate_id: (
                            bootstrap_progress.advance(
                                detail=f"{candidate_id} | {completed}/{total}"
                            )
                        )
                    )
                    if bootstrap_progress is not None
                    else None
                ),
            )
        except (Exception, KeyboardInterrupt) as exc:
            if bootstrap_progress is not None:
                bootstrap_progress.fail(exc)
            raise
        candidate_tuning = tuned.loc[
            tuned["candidate_id"] == candidate.spec.candidate_id
        ]
        parameter_counts = Counter(candidate_tuning["best_parameters_json"])
        frequent_parameters, frequent_count = parameter_counts.most_common(1)[0]
        summaries.append(
            {
                "experiment_id": experiment_id,
                "dataset_id": config.schema.dataset_id,
                "target": target_name,
                "task_kind": task.config.kind,
                "candidate_id": candidate.spec.candidate_id,
                "candidate_order": candidate.spec.order,
                "input_id": candidate.spec.dataset.input_id,
                "input_mode": candidate.spec.input_mode,
                "representation": candidate.spec.representation,
                "model": candidate.spec.model_name,
                "evaluation_unit": config.selection.evaluation_unit,
                "total_candidate_count": len(candidate_specs),
                "common_trial_count": len(common_trials),
                "subjects_total": len(subjects),
                "outer_folds_run": len(outer_subjects),
                "complete_loso": len(outer_subjects) == len(subjects),
                "primary_metric": (
                    config.selection.classification_metric
                    if task.is_classification
                    else config.selection.regression_metric
                ),
                "primary_score": subject_primary,
                "primary_evaluable_subjects": evaluable_subjects,
                "primary_excluded_subjects": excluded_subjects,
                "best_parameter_counts_json": json.dumps(
                    dict(parameter_counts), sort_keys=True
                ),
                "most_frequent_parameters_json": frequent_parameters,
                "parameter_stability_fraction": (
                    frequent_count / len(candidate_tuning)
                ),
                "quick_grid": config.evaluation.quick,
                "inner_fit_seconds": float(
                    sum(
                        row["inner_fit_seconds"]
                        for row in fold_rows
                        if row["candidate_id"] == candidate.spec.candidate_id
                    )
                ),
                "outer_fit_seconds": float(
                    sum(
                        row["fit_seconds"]
                        for row in fold_rows
                        if row["candidate_id"] == candidate.spec.candidate_id
                    )
                ),
                "target_run_seconds": time.perf_counter() - started,
                **pooled,
                **intervals,
            }
        )

    if bootstrap_progress is not None:
        bootstrap_progress.finish()

    return NestedTargetResult(
        summaries=summaries,
        folds=fold_rows,
        predictions=prediction_rows,
        inner_candidate_scores=inner_rows,
        inner_split_assignments=split_assignment_rows,
        tuned_candidates=[],
        window_predictions=window_rows,
        permutation_importance=permutation_records,
        shap_importance=shap_records,
        shap_values=shap_value_records,
    )
