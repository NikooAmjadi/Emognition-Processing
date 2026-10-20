"""Edit this file to configure an experiment; no command-line parameters are used."""

from pathlib import Path

from thesis_ml.config_types import (
    AnalysisConfig,
    AppConfig,
    CandidateConfig,
    DataConfig,
    DatasetSchema,
    EvaluationConfig,
    ModelConfig,
    NestedSelectionConfig,
    OutputConfig,
    PathsConfig,
    PermutationImportanceConfig,
    ShapConfig,
    TaskConfig,
)

PROJECT_DIR = Path(__file__).resolve().parent

# To use another dataset, change the paths and schema below. The training,
# evaluation, explainability, and reporting modules do not contain dataset names.
PATHS = PathsConfig(
    input_files=(PROJECT_DIR / "features" / "emognition_features_w60_o30.csv",),
    input_globs=(),
    output_dir=PROJECT_DIR / "model_results",
    run_name=None,
    # Keep this None for a new run. After an interruption, paste the existing
    # run folder here; its configuration and input hashes must still match.
    # Example: PROJECT_DIR / "model_results" / "20261008T181957Z_173b798590"
    resume_from=None,
)

SCHEMA = DatasetSchema(
    dataset_id="physiological_features",
    subject_column="subject_id",
    condition_column="emotion",
    trial_id_columns=("subject_id", "movie_order_index", "emotion"),
    window_id_column="window_id",
    aggregation_columns=("subject_id", "emotion", "movie_order_index"),
    target_columns={"valence": "sam_valence", "arousal": "sam_arousal"},
    feature_prefixes={"hc": "hc_", "nvg": "nvg_", "hvg": "hvg_"},
    representations={
        "hc": ("hc",),
        "nvg": ("nvg",),
        "hvg": ("hvg",),
        "hc+nvg": ("hc", "nvg"),
        "hc+hvg": ("hc", "hvg"),
    },
    expected_feature_counts={"hc": 29, "nvg": 48, "hvg": 48},
    excluded_conditions=("BASELINE", "NEUTRAL"),
    graph_node_tokens=("bvp", "eda", "temp"),
    graph_node_aliases={"ppg": "bvp", "skt": "temp"},
)

DATA = DataConfig(
    trial_policy="common",
    min_trials_per_subject=6,
    input_modes=("windowed", "aggregated"),
)

# To change the analysis type, change only kind to "binary", "ternary",
# "multiclass", or "regression". Their fixed definitions live in
# thesis_ml/task_presets.py and are recorded in every run manifest.
TASK = TaskConfig(
    kind="regression",
    targets=("valence", "arousal"),
)

MODELS = ModelConfig(
    enabled=("gcn",),#"svm", "rf", "knn", "xgboost", "mlp", "gcn", "gat"),
    representations=("hc", "nvg", "hvg", "hc+nvg", "hc+hvg"),
    parameter_overrides={},
    graph_representations=("hc", "nvg", "hvg", "hc+nvg", "hc+hvg"),
    device="cpu",
    graph_device="auto",
    graph_max_epochs=30,
    graph_min_epochs=5,
    graph_patience=8,
)

QUICK_RUN = False

EVALUATION = EvaluationConfig(
    seed=42,
    n_jobs=1,
    inner_splits=3,
    # Development runs skip confidence intervals. Changing QUICK_RUN to False
    # automatically restores 1,000 subject-cluster bootstrap repetitions.
    bootstrap_repeats=0 if QUICK_RUN else 1000,
    quick=QUICK_RUN,
    max_outer_folds=0,
    show_progress=True,
)

# Candidate space is declared before results are observed. input_id="*" expands
# each template to every configured input CSV. Reduce these tuples for a focused,
# confirmatory thesis analysis; the complete space can be computationally large.
CANDIDATES = tuple(
    CandidateConfig(
        name=f"{model}__{mode}__{representation}",
        input_id="*",
        input_mode=mode,
        representation=representation,
        model_name=model,
    )
    for mode in DATA.input_modes
    for representation in MODELS.representations
    for model in MODELS.enabled
    if model not in {"gcn", "gat"}
    or representation in MODELS.graph_representations
)

SELECTION = NestedSelectionConfig(
    candidates=CANDIDATES,
    evaluation_unit="trial",
    classification_metric="subject_class_macro_recall",
    regression_metric="subject_macro_mae",
)

ANALYSES = AnalysisConfig(
    shap=ShapConfig(
        enabled=False,
        models=("xgboost",),
        background_size=100,
        max_test_samples=0,
        top_k=10,
        save_local_values=False,
    ),
    permutation_importance=PermutationImportanceConfig(
        enabled=False,
        models=("xgboost",),
        repeats=10,
    ),
    correlations=False,
    class_distribution=False,
    statistical_tests=True,
    statistical_permutation_repeats=10_000,
)

OUTPUT = OutputConfig(
    save_summary=True,
    save_subject_results=True,
    save_trial_predictions=True,
    # Large diagnostic artifacts are opt-in.
    save_window_predictions=False,
    save_inner_search_details=False,
    save_split_assignments=False,
    save_explainability_fold_details=False,
    # One transactional SQLite file holds all resumable state.
    save_checkpoints=True,
    # It is automatically removed only after a successful completed run.
    keep_checkpoint_after_completion=False,
)

CONFIG = AppConfig(
    paths=PATHS,
    schema=SCHEMA,
    data=DATA,
    task=TASK,
    models=MODELS,
    evaluation=EVALUATION,
    selection=SELECTION,
    analyses=ANALYSES,
    output=OUTPUT,
)
