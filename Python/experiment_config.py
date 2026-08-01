from config import Config


experiment_config = Config(
    raw_dataset_path="emognition_raw_features_60.csv",
    vg_dataset_path="emognition_vg_features_60.csv",

    feature_modes=[
        "vg","rawvg"
    ],

    input_modes=[
        "windowed", "aggregated"
    ],

    target_label="valence",
    classification="binary",

    models_to_run=[
        "GAT", "MLP", "GCN"
    ],

    fast_mode=True,
    normalize_labels=True,
    remove_baseline_neutral=True,

    random_state=42,
    n_jobs_search=1,
    n_jobs_xgb=-1,
    inner_cv_splits=3,
    scoring="f1_macro",

    graph_device="auto",
    graph_batch_size=32,
    graph_eval_batch_size=128,
    graph_max_epochs=100,
    graph_patience=15,
)
