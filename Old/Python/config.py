from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Config:
    raw_dataset_path: str = "emognition_raw_features_60.csv"
    vg_dataset_path: str = "emognition_vg_features_60.csv"

    subject_col: str = "subject_id"
    emotion_col: str = "emotion"
    valence_col: str = "label_valence"
    arousal_col: str = "label_arousal"

    window_columns: list[str] = field(
        default_factory=lambda: [
            "window_id",
            "window_start_sec",
            "window_end_sec",
            "window_size_sec",
            "overlap_sec",
        ]
    )

    feature_modes: list[str] = field(
        default_factory=lambda: ["raw", "vg", "rawvg"]
    )
    input_modes: list[str] = field(
        default_factory=lambda: ["aggregated", "windowed"]
    )

    target_label: str = "arousal"       # valence | arousal
    classification: str = "regression"  # binary | ternary | regression

    models_to_run: list[str] = field(
        default_factory=lambda: [
            "SVM",
            "Random Forest",
            "KNN",
            "XGBoost",
        ]
    )

    fast_mode: bool = False
    normalize_labels: bool = False
    remove_baseline_neutral: bool = True

    random_state: int = 42
    n_jobs_search: int = 1
    n_jobs_xgb: int = -1
    xgb_device: str = "cpu"  # cpu | cuda
    inner_cv_splits: int = 3
    scoring: str = "f1_macro"

    # Shared graph-training settings.
    graph_device: str = "auto"  # auto | cpu | cuda
    graph_batch_size: int = 32
    graph_eval_batch_size: int = 128
    graph_max_epochs: int = 100
    graph_patience: int = 15
    graph_min_delta: float = 1e-5
    graph_num_workers: int = 0
    graph_verbose: bool = True

    # The graph is one node per physiological modality. PyG layers add
    # self-loops internally by default, so explicit self-edges are disabled.
    graph_include_self_edges: bool = False
    graph_allowed_feature_modes: list[str] = field(
        default_factory=lambda: ["vg", "rawvg"]
    )
    graph_node_prefixes: dict[str, list[str]] = field(
        default_factory=lambda: {
            "vg": [
                "BVP_vg_",
                "EDA_vg_",
                "TEMP_vg_",
            ],
            "rawvg": [
                "vg_BVP_vg_",
                "vg_EDA_vg_",
                "vg_TEMP_vg_",
            ],
        }
    )

    # Parameter grids are intentionally kept in Config so a single experiment
    # file fully describes the graph search space.
    gcn_param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "hidden_channels": [32, 64],
            "dropout": [0.2, 0.5],
            "learning_rate": [0.001],
            "weight_decay": [0.0, 5e-4],
        }
    )
    gat_param_grid: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "hidden_channels": [32, 64],
            "heads": [2, 4],
            "dropout": [0.2, 0.5],
            "learning_rate": [0.001],
            "weight_decay": [0.0, 5e-4],
        }
    )

    output_summary_template: str = (
        "simulation_results_{target}_{input_type}_{mode}_"
        "{class_name}_{norm}_{filter}_{is_fast}.csv"
    )
    output_subjects_template: str = (
        "subjects_results_{target}_{input_type}_{mode}_"
        "{class_name}_{norm}_{filter}_{is_fast}.csv"
    )

    def __post_init__(self) -> None:
        valid_target_labels = {"valence", "arousal"}
        valid_classifications = {"binary", "ternary", "regression"}
        valid_feature_modes = {"raw", "vg", "rawvg"}
        valid_input_modes = {"aggregated", "windowed"}
        valid_graph_devices = {"auto", "cpu", "cuda"}
        valid_xgb_devices = {"cpu", "cuda"}

        if self.target_label not in valid_target_labels:
            raise ValueError(
                "target_label must be one of "
                f"{sorted(valid_target_labels)}. Received: {self.target_label!r}"
            )

        if self.classification not in valid_classifications:
            raise ValueError(
                "classification must be one of "
                f"{sorted(valid_classifications)}. "
                f"Received: {self.classification!r}"
            )

        self._validate_non_empty_values(
            "feature_modes", self.feature_modes
        )
        self._validate_non_empty_values(
            "input_modes", self.input_modes
        )
        self._validate_non_empty_values(
            "models_to_run", self.models_to_run
        )
        self._validate_non_empty_values(
            "window_columns", self.window_columns
        )

        invalid_feature_modes = sorted(
            set(self.feature_modes) - valid_feature_modes
        )
        if invalid_feature_modes:
            raise ValueError(
                f"Invalid feature_modes: {invalid_feature_modes}. "
                f"Allowed values: {sorted(valid_feature_modes)}"
            )

        invalid_input_modes = sorted(
            set(self.input_modes) - valid_input_modes
        )
        if invalid_input_modes:
            raise ValueError(
                f"Invalid input_modes: {invalid_input_modes}. "
                f"Allowed values: {sorted(valid_input_modes)}"
            )

        if len(set(self.models_to_run)) != len(self.models_to_run):
            raise ValueError("models_to_run cannot contain duplicate names.")

        if self.inner_cv_splits < 2:
            raise ValueError("inner_cv_splits must be at least 2.")
        if self.n_jobs_search == 0:
            raise ValueError("n_jobs_search cannot be zero.")
        if self.n_jobs_xgb == 0:
            raise ValueError("n_jobs_xgb cannot be zero.")
        if self.xgb_device not in valid_xgb_devices:
            raise ValueError("xgb_device must be 'cpu' or 'cuda'.")
        if not self.scoring.strip():
            raise ValueError("scoring cannot be empty.")

        if self.graph_device not in valid_graph_devices:
            raise ValueError(
                "graph_device must be 'auto', 'cpu', or 'cuda'."
            )
        if self.graph_batch_size < 1:
            raise ValueError("graph_batch_size must be at least 1.")
        if self.graph_eval_batch_size < 1:
            raise ValueError("graph_eval_batch_size must be at least 1.")
        if self.graph_max_epochs < 1:
            raise ValueError("graph_max_epochs must be at least 1.")
        if self.graph_patience < 1:
            raise ValueError("graph_patience must be at least 1.")
        if self.graph_min_delta < 0:
            raise ValueError("graph_min_delta cannot be negative.")
        if self.graph_num_workers < 0:
            raise ValueError("graph_num_workers cannot be negative.")

        invalid_graph_modes = sorted(
            set(self.graph_allowed_feature_modes) - valid_feature_modes
        )
        if invalid_graph_modes:
            raise ValueError(
                "graph_allowed_feature_modes contains invalid values: "
                f"{invalid_graph_modes}"
            )

        for mode in self.graph_allowed_feature_modes:
            prefixes = self.graph_node_prefixes.get(mode)
            if not prefixes:
                raise ValueError(
                    f"graph_node_prefixes must define at least one prefix "
                    f"for graph feature mode {mode!r}."
                )
            if len(set(prefixes)) != len(prefixes):
                raise ValueError(
                    f"Duplicate graph node prefixes found for mode {mode!r}."
                )

        self._validate_param_grid("gcn_param_grid", self.gcn_param_grid)
        self._validate_param_grid("gat_param_grid", self.gat_param_grid)

    @staticmethod
    def _validate_non_empty_values(name: str, values: list[str]) -> None:
        if not values:
            raise ValueError(f"{name} cannot be empty.")
        if any(not str(value).strip() for value in values):
            raise ValueError(f"{name} cannot contain empty values.")

    @staticmethod
    def _validate_param_grid(
        name: str,
        grid: dict[str, list[Any]],
    ) -> None:
        if not grid:
            raise ValueError(f"{name} cannot be empty.")
        empty_parameters = [
            parameter
            for parameter, values in grid.items()
            if not values
        ]
        if empty_parameters:
            raise ValueError(
                f"{name} contains parameters without values: "
                f"{empty_parameters}"
            )

    @property
    def label_col(self) -> str:
        if self.target_label == "arousal":
            return self.arousal_col
        if self.target_label == "valence":
            return self.valence_col
        raise ValueError(
            f"Unsupported target_label: {self.target_label!r}"
        )

    @property
    def all_label_cols(self) -> list[str]:
        return [self.valence_col, self.arousal_col]

    @property
    def group_cols(self) -> list[str]:
        return [self.subject_col, self.emotion_col, self.label_col]

    @property
    def scoring_metric(self) -> str:
        if self.classification == "regression":
            return "neg_mean_absolute_error"
        return self.scoring

    @property
    def output_channels(self) -> int:
        if self.classification == "binary":
            return 2
        if self.classification == "ternary":
            return 3
        return 1

    def graph_prefixes_for(self, feature_mode: str) -> list[str]:
        if feature_mode not in self.graph_allowed_feature_modes:
            raise ValueError(
                f"Graph models do not support feature_mode={feature_mode!r}. "
                "Allowed graph modes: "
                f"{self.graph_allowed_feature_modes}"
            )

        prefixes = self.graph_node_prefixes.get(feature_mode)
        if not prefixes:
            raise ValueError(
                "No graph node prefixes are configured for "
                f"feature_mode={feature_mode!r}."
            )
        return list(prefixes)
