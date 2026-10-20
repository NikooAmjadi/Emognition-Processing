from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..config_types import AppConfig
from ..domain import Dataset, RunResults
from ..errors import ExperimentError
from ..task_presets import get_task_preset
from ..utils import atomic_write_csv, atomic_write_json, json_default, safe_token


def _dependency_versions() -> dict[str, str | None]:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "shap", "torch", "torch-geometric", "xgboost")
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _analysis_contract(config: AppConfig) -> dict[str, Any]:
    classification = config.task.kind != "regression"
    task_preset = get_task_preset(config.task.preset_set, config.task.kind)
    return {
        "dataset_adapter": config.schema.adapter,
        "task_kind": config.task.kind,
        "task_preset_set": config.task.preset_set,
        "target_mapping": (
            task_preset.class_groups if classification else "numeric target without class binning"
        ),
        "excluded_target_values": task_preset.excluded_values,
        "outer_evaluation": "LeaveOneGroupOut by configured subject column",
        "inner_selection": (
            "Every predeclared candidate is independently tuned inside the same "
            "grouped folds, then every tuned candidate is evaluated on every outer "
            "LOSO subject; stratified when classification permits"
        ),
        "primary_evaluation_unit": (
            "Trial-level predictions with equal subject weighting; window predictions "
            "are aggregated before candidate comparison"
        ),
        "preprocessing": "Training-fold median imputation; zero-variance removal for classical tabular models; scaling where model requires it",
        "aggregation": "Per-trial population mean and population standard deviation",
        "graph_models": (
            "Interim sensor-descriptor graph: one BVP, EDA, and TEMP node; "
            "aligned HC/NVG/HVG descriptor blocks with presence masks; complete "
            "directed inter-sensor edges. This is not the raw visibility topology."
        ),
        "confidence_intervals": (
            "Subject-cluster percentile bootstrap, including the primary score"
        ),
        "statistical_tests": "Paired subject-cluster permutation with Holm correction",
        "importance": "Permutation importance on untouched outer-test data",
        "shap": "SHAP explanation on untouched outer-test data using outer-training background only",
    }


def configuration_hash(config: AppConfig, datasets: list[Dataset]) -> str:
    task_preset = get_task_preset(config.task.preset_set, config.task.kind)
    configuration = asdict(config)
    # Filesystem locations are operational state. Resolved input content is
    # protected separately by the ordered dataset hashes below, allowing a run
    # to resume after (for example) a Colab drive is remounted elsewhere.
    configuration.pop("paths", None)
    # Console rendering does not alter the experiment definition.
    configuration["evaluation"].pop("show_progress", None)
    payload = {
        "config": configuration,
        "resolved_task_preset": asdict(task_preset),
        "inputs": [dataset.sha256 for dataset in datasets],
    }
    serialized = json.dumps(payload, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:10]


class OutputManager:
    def __init__(self, config: AppConfig, datasets: list[Dataset]):
        fingerprint = configuration_hash(config, datasets)
        self.config = config
        self.fingerprint = fingerprint
        self.resumed = config.paths.resume_from is not None
        self.previous_manifest: dict[str, Any] = {}
        if self.resumed:
            self.run_dir = config.paths.resume_from.resolve()
            manifest_path = self.run_dir / "run_manifest.json"
            if not self.run_dir.is_dir() or not manifest_path.is_file():
                raise ExperimentError(
                    f"Resume directory has no run_manifest.json: {self.run_dir}"
                )
            try:
                self.previous_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ExperimentError(
                    f"Cannot read resume manifest: {manifest_path}"
                ) from exc
            previous_hash = self.previous_manifest.get("configuration_hash")
            if previous_hash != fingerprint:
                raise ExperimentError(
                    "Resume configuration or input data do not match the original "
                    f"run ({previous_hash!r} != {fingerprint!r})"
                )
            if self.previous_manifest.get("status") == "complete":
                raise ExperimentError(f"Run is already complete: {self.run_dir}")
        else:
            name = (
                config.paths.run_name
                or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")
                + fingerprint
            )
            parent = config.paths.output_dir.resolve()
            self.run_dir = parent / safe_token(name)
            if self.run_dir.exists():
                raise ExperimentError(f"Run directory already exists: {self.run_dir}")
            self.run_dir.mkdir(parents=True)

    def write_manifest(
        self,
        status: str,
        datasets: list[Dataset],
        **extra: Any,
    ) -> None:
        if not self.config.output.save_manifest:
            return
        payload = {
            "framework": "thesis_ml",
            "framework_version": "3.0.0",
            "status": status,
            "configuration_hash": self.fingerprint,
            "configuration": asdict(self.config),
            "resolved_task_preset": asdict(
                get_task_preset(
                    self.config.task.preset_set, self.config.task.kind
                )
            ),
            "analysis_contract": _analysis_contract(self.config),
            "dependencies": _dependency_versions(),
            "inputs": [
                {
                    "dataset_id": dataset.dataset_id,
                    "input_id": dataset.input_id,
                    "path_hint": dataset.path.name,
                    "sha256": dataset.sha256,
                    "rows": len(dataset.table),
                    "subjects": dataset.table[self.config.schema.subject_column].nunique(),
                    "trials": dataset.table["trial_id"].nunique(),
                    "metadata": dataset.metadata,
                }
                for dataset in datasets
            ],
            **extra,
        }
        atomic_write_json(payload, self.run_dir / "run_manifest.json")

    def write_eligibility(self, participants: list[dict[str, Any]], inputs: list[dict[str, Any]]) -> None:
        atomic_write_csv(pd.DataFrame(participants), self.run_dir / "participant_eligibility.csv")
        atomic_write_csv(pd.DataFrame(inputs), self.run_dir / "input_eligibility.csv")

    def write_results(self, results: RunResults, progress: dict[str, Any]) -> None:
        output = self.config.output
        if output.save_summary:
            summary = pd.DataFrame(results.summaries)
            if not summary.empty:
                summary["primary_rank_within_target"] = summary.groupby(
                    ["target", "task_kind"], dropna=False
                )["primary_score"].rank(
                    ascending=self.config.task.kind == "regression",
                    method="min",
                )
            atomic_write_csv(summary, self.run_dir / "summary.csv")
        if output.save_subject_results:
            folds = pd.DataFrame(results.folds)
            atomic_write_csv(folds, self.run_dir / "subject_results.csv")
        if output.save_trial_predictions:
            predictions = pd.DataFrame(results.predictions)
            atomic_write_csv(predictions, self.run_dir / "oof_trial_predictions.csv")
        if output.save_window_predictions and results.window_predictions:
            atomic_write_csv(
                pd.DataFrame(results.window_predictions),
                self.run_dir / "oof_window_predictions.csv",
            )
        if output.save_inner_search_details and results.inner_candidate_scores:
            atomic_write_csv(
                pd.DataFrame(results.inner_candidate_scores),
                self.run_dir / "inner_candidate_scores.csv",
            )
        if output.save_split_assignments and results.inner_split_assignments:
            atomic_write_csv(
                pd.DataFrame(results.inner_split_assignments),
                self.run_dir / "inner_split_assignments.csv",
            )
        if output.save_permutation_importance and results.permutation_importance:
            frame = pd.DataFrame(results.permutation_importance)
            if output.save_explainability_fold_details:
                atomic_write_csv(
                    frame, self.run_dir / "permutation_importance_folds.csv"
                )
            group = [column for column in ("experiment_id", "dataset_id", "input_id", "target", "task_kind", "input_mode", "representation", "model", "feature", "source_feature") if column in frame]
            summary = frame.groupby(group, dropna=False)["importance_mean"].agg(mean_importance="mean", std_importance="std", evaluated_folds="count").reset_index()
            summary["rank"] = summary.groupby("experiment_id")["mean_importance"].rank(ascending=False, method="min")
            atomic_write_csv(summary, self.run_dir / "permutation_importance_summary.csv")
        if output.save_shap and results.shap_importance:
            frame = pd.DataFrame(results.shap_importance)
            if output.save_explainability_fold_details:
                atomic_write_csv(frame, self.run_dir / "shap_importance_folds.csv")
            group = [column for column in ("experiment_id", "dataset_id", "input_id", "target", "task_kind", "input_mode", "representation", "model", "output", "feature", "source_feature") if column in frame]
            summary = frame.groupby(group, dropna=False).agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"), mean_rank=("rank_within_fold", "mean"), evaluated_folds=("fold", "nunique")).reset_index()
            summary["rank"] = summary.groupby(["experiment_id", "output"])["mean_abs_shap"].rank(ascending=False, method="min")
            atomic_write_csv(summary, self.run_dir / "shap_importance_summary.csv")
        if (
            output.save_shap
            and self.config.analyses.shap.save_local_values
            and results.shap_values
        ):
            atomic_write_csv(pd.DataFrame(results.shap_values), self.run_dir / "shap_values.csv")

    def finalize_checkpoint(self) -> None:
        if (
            self.config.output.save_checkpoints
            and not self.config.output.keep_checkpoint_after_completion
        ):
            checkpoint = self.run_dir / "checkpoint.sqlite"
            if checkpoint.is_file():
                checkpoint.unlink()
