from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ExperimentError
from .task_presets import get_task_preset

TaskKind = Literal["binary", "ternary", "multiclass", "regression"]


@dataclass(frozen=True)
class PathsConfig:
    input_files: tuple[Path, ...] = ()
    input_globs: tuple[str, ...] = ("features/*.csv",)
    output_dir: Path = Path("model_results")
    run_name: str | None = None
    resume_from: Path | None = None


@dataclass(frozen=True)
class DatasetSchema:
    """All dataset-specific knowledge is kept in this object."""

    dataset_id: str = "physiological_features"
    adapter: str = "generic_csv"
    subject_column: str = "subject_id"
    condition_column: str | None = "emotion"
    trial_id_columns: tuple[str, ...] = (
        "subject_id",
        "movie_order_index",
        "emotion",
    )
    window_id_column: str | None = "window_id"
    aggregation_columns: tuple[str, ...] = (
        "subject_id",
        "emotion",
        "movie_order_index",
    )
    pass_through_columns: tuple[str, ...] = (
        "emotion",
        "movie_order_index",
        "window_id",
        "window_start_sec",
        "window_end_sec",
    )
    target_columns: dict[str, str] = field(
        default_factory=lambda: {
            "valence": "sam_valence",
            "arousal": "sam_arousal",
        }
    )
    feature_prefixes: dict[str, str] = field(
        default_factory=lambda: {"hc": "hc_", "nvg": "nvg_", "hvg": "hvg_"}
    )
    representations: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "hc": ("hc",),
            "nvg": ("nvg",),
            "hvg": ("hvg",),
            "hc+nvg": ("hc", "nvg"),
            "hc+hvg": ("hc", "hvg"),
        }
    )
    expected_feature_counts: dict[str, int] | None = field(
        default_factory=lambda: {"hc": 29, "nvg": 48, "hvg": 48}
    )
    excluded_conditions: tuple[str, ...] = ("BASELINE", "NEUTRAL")
    condition_casefold: bool = True
    numeric_target_min: float | None = 1
    numeric_target_max: float | None = 9
    manifest_validation: bool = True
    graph_node_tokens: tuple[str, ...] = ("bvp", "eda", "temp")
    graph_node_aliases: dict[str, str] = field(
        default_factory=lambda: {"ppg": "bvp", "skt": "temp"}
    )


@dataclass(frozen=True)
class DataConfig:
    trial_policy: Literal["common", "available"] = "common"
    min_trials_per_subject: int = 6
    input_modes: tuple[Literal["windowed", "aggregated"], ...] = (
        "windowed",
        "aggregated",
    )


@dataclass(frozen=True)
class TaskConfig:
    kind: TaskKind = "binary"
    targets: tuple[str, ...] = ("valence", "arousal")
    preset_set: str = "sam_1_9"


@dataclass(frozen=True)
class ModelConfig:
    enabled: tuple[str, ...] = ("svm", "rf", "knn", "xgboost", "mlp", "gcn", "gat")
    representations: tuple[str, ...] = ("hc", "nvg", "hvg", "hc+nvg", "hc+hvg")
    parameter_overrides: dict[str, dict[str, tuple[Any, ...]]] = field(default_factory=dict)
    graph_representations: tuple[str, ...] = (
        "hc",
        "nvg",
        "hvg",
        "hc+nvg",
        "hc+hvg",
    )
    device: Literal["cpu", "cuda"] = "cpu"
    graph_device: Literal["auto", "cpu", "cuda"] = "auto"
    graph_batch_size: int = 32
    graph_eval_batch_size: int = 128
    graph_max_epochs: int = 100
    graph_min_epochs: int = 5
    graph_patience: int = 30
    graph_min_delta: float = 1e-5


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 42
    n_jobs: int = 1
    inner_splits: int = 3
    bootstrap_repeats: int = 1000
    quick: bool = False
    max_outer_folds: int = 0
    show_progress: bool = True


@dataclass(frozen=True)
class CandidateConfig:
    """One predeclared pipeline tuned inside inner CV and tested in outer LOSO."""

    name: str
    input_id: str
    input_mode: Literal["windowed", "aggregated"]
    representation: str
    model_name: str


@dataclass(frozen=True)
class NestedSelectionConfig:
    candidates: tuple[CandidateConfig, ...] = ()
    evaluation_unit: Literal["trial"] = "trial"
    classification_metric: Literal["subject_class_macro_recall"] = (
        "subject_class_macro_recall"
    )
    regression_metric: Literal["subject_macro_mae"] = "subject_macro_mae"


@dataclass(frozen=True)
class ShapConfig:
    enabled: bool = True
    models: tuple[str, ...] = ("xgboost",)
    background_size: int = 100
    max_test_samples: int = 0
    top_k: int = 10
    save_local_values: bool = False


@dataclass(frozen=True)
class PermutationImportanceConfig:
    enabled: bool = True
    models: tuple[str, ...] = ("xgboost",)
    repeats: int = 10


@dataclass(frozen=True)
class AnalysisConfig:
    shap: ShapConfig = field(default_factory=ShapConfig)
    permutation_importance: PermutationImportanceConfig = field(
        default_factory=PermutationImportanceConfig
    )
    correlations: bool = True
    class_distribution: bool = True
    statistical_tests: bool = True
    statistical_permutation_repeats: int = 10_000


@dataclass(frozen=True)
class OutputConfig:
    save_summary: bool = True
    save_subject_results: bool = True
    save_trial_predictions: bool = True
    save_window_predictions: bool = False
    save_inner_search_details: bool = False
    save_split_assignments: bool = False
    save_manifest: bool = True
    save_permutation_importance: bool = True
    save_shap: bool = True
    save_explainability_fold_details: bool = False
    save_descriptive_reports: bool = True
    save_checkpoints: bool = True
    keep_checkpoint_after_completion: bool = False


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    schema: DatasetSchema = field(default_factory=DatasetSchema)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    selection: NestedSelectionConfig = field(default_factory=NestedSelectionConfig)
    analyses: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if not self.schema.dataset_id.strip():
            raise ExperimentError("schema.dataset_id cannot be empty")
        if not self.schema.adapter.strip():
            raise ExperimentError("schema.adapter cannot be empty")
        if self.data.trial_policy not in {"common", "available"}:
            raise ExperimentError("data.trial_policy must be 'common' or 'available'")
        if self.data.min_trials_per_subject < 0:
            raise ExperimentError("data.min_trials_per_subject cannot be negative")
        if self.task.kind not in {"binary", "ternary", "multiclass", "regression"}:
            raise ExperimentError(f"Unknown task kind: {self.task.kind}")
        task_preset = get_task_preset(self.task.preset_set, self.task.kind)
        if self.task.kind == "binary" and len(task_preset.class_groups) != 2:
            raise ExperimentError("A binary task must define exactly two class_groups")
        if self.task.kind == "ternary" and len(task_preset.class_groups) != 3:
            raise ExperimentError("A ternary task must define exactly three class_groups")
        if self.task.kind != "regression" and len(task_preset.class_groups) < 2:
            raise ExperimentError("Classification requires at least two class_groups")
        flattened = [
            value for values in task_preset.class_groups.values() for value in values
        ]
        if len(flattened) != len(set(flattened)):
            raise ExperimentError("Values in task.class_groups must not overlap")
        if (
            task_preset.positive_class is not None
            and task_preset.positive_class not in task_preset.class_groups
        ):
            raise ExperimentError("positive_class must name one of the preset class_groups")
        unknown_targets = set(self.task.targets) - set(self.schema.target_columns)
        if unknown_targets:
            raise ExperimentError(f"Targets missing from schema.target_columns: {sorted(unknown_targets)}")
        unknown_representations = set(self.models.representations) - set(self.schema.representations)
        if unknown_representations:
            raise ExperimentError(f"Unknown representations: {sorted(unknown_representations)}")
        unknown_graph_representations = set(
            self.models.graph_representations
        ) - set(self.schema.representations)
        if unknown_graph_representations:
            raise ExperimentError(
                "Unknown graph representations: "
                f"{sorted(unknown_graph_representations)}"
            )
        if self.models.graph_representations:
            unknown_alias_targets = set(
                self.schema.graph_node_aliases.values()
            ) - set(self.schema.graph_node_tokens)
            if unknown_alias_targets:
                raise ExperimentError(
                    "graph_node_aliases target unknown nodes: "
                    f"{sorted(unknown_alias_targets)}"
                )
        if self.evaluation.inner_splits < 2:
            raise ExperimentError("evaluation.inner_splits must be at least 2")
        if self.evaluation.n_jobs == 0:
            raise ExperimentError("evaluation.n_jobs cannot be zero")
        if self.evaluation.max_outer_folds < 0:
            raise ExperimentError("evaluation.max_outer_folds cannot be negative")
        if self.evaluation.bootstrap_repeats < 0:
            raise ExperimentError("evaluation.bootstrap_repeats cannot be negative")
        if self.paths.resume_from is not None and not self.output.save_checkpoints:
            raise ExperimentError(
                "output.save_checkpoints must be enabled when paths.resume_from is set"
            )
        if not self.selection.candidates:
            raise ExperimentError("selection.candidates cannot be empty")
        candidate_names = [candidate.name for candidate in self.selection.candidates]
        if len(candidate_names) != len(set(candidate_names)):
            raise ExperimentError("selection candidate names must be unique")
        for candidate in self.selection.candidates:
            if not candidate.name.strip():
                raise ExperimentError("selection candidate names cannot be empty")
            if candidate.input_mode not in {"windowed", "aggregated"}:
                raise ExperimentError(f"Invalid candidate input mode: {candidate.input_mode}")
            if candidate.representation not in self.schema.representations:
                raise ExperimentError(
                    f"Candidate {candidate.name!r} uses unknown representation "
                    f"{candidate.representation!r}"
                )
        if self.analyses.shap.background_size < 1 or self.analyses.shap.top_k < 1:
            raise ExperimentError("SHAP background_size and top_k must be positive")
        if (
            self.analyses.statistical_tests
            and self.analyses.statistical_permutation_repeats < 1
        ):
            raise ExperimentError(
                "statistical_permutation_repeats must be positive when tests are enabled"
            )
        if self.models.graph_min_epochs > self.models.graph_max_epochs:
            raise ExperimentError("graph_min_epochs cannot exceed graph_max_epochs")
