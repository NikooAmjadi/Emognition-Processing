import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV, ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")

# raw | vg | rawvg
FEATURE_MODE = ["raw", "vg", "rawvg"]

# windowed | aggregated
INPUT_MODE = ["aggregated", "windowed"]

# valence | arousal
TARGET_LABEL = "arousal"

# binary | ternary | regression
CLASSIFICATION = "ternary"

# fast mode disables grid search
FAST_MODE = False

# Apply Subject-wise Z-score normalization to self-assessment labels
NORMALIZE_LABELS = True

# Remove baseline and neutral trials
REMOVE_BASELINE_NEUTRAL = True

RAW_DATASET_PATH = "emognition_raw_features_60.csv"
VG_DATASET_PATH = "emognition_vg_features_60.csv"

SUBJECT_COL = "subject_id"
EMOTION_COL = "emotion"
VALENCE_COL = "label_valence"
AROUSAL_COL = "label_arousal"

LABEL_COL = {"valence": VALENCE_COL,"arousal": AROUSAL_COL}[TARGET_LABEL]

WINDOW_COLUMNS = [
    "window_id",
    "window_start_sec",
    "window_end_sec",
    "window_size_sec",
    "overlap_sec"
]

GROUP_COLS = [SUBJECT_COL, EMOTION_COL, LABEL_COL]
ALL_LABEL_COLS = [VALENCE_COL, AROUSAL_COL]

OUTPUT_SUMMARY_PATH = (
    "simulation_results_{target}_{input_type}_{mode}_{class_name}_{norm}_{filter}_{is_fast}.csv"
)

OUTPUT_SUBJECTS_PATH = (
    "subjects_results_{target}_{input_type}_{mode}_{class_name}_{norm}_{filter}_{is_fast}.csv"
)

RANDOM_STATE = 42
N_JOBS_SEARCH = 1
N_JOBS_XGB = -1

INNER_CV_SPLITS = 3
SCORING = "f1_macro"


def load_dataset(input_type):
    if input_type == "raw":
        return pd.read_csv(RAW_DATASET_PATH)

    if input_type == "vg":
        return pd.read_csv(VG_DATASET_PATH)

    if input_type == "rawvg":
        raw_df = pd.read_csv(RAW_DATASET_PATH)
        vg_df = pd.read_csv(VG_DATASET_PATH)

        align_cols = [SUBJECT_COL] + WINDOW_COLUMNS

        if not raw_df[align_cols].equals(vg_df[align_cols]):
            raise ValueError("Raw and VG files are not aligned.")

        exclude_cols = GROUP_COLS + WINDOW_COLUMNS + ALL_LABEL_COLS

        raw_feature_cols = [
            col for col in raw_df.columns
            if col not in exclude_cols
        ]

        vg_feature_cols = [
            col for col in vg_df.columns
            if col not in exclude_cols
        ]

        meta_cols = GROUP_COLS + WINDOW_COLUMNS

        raw_features = raw_df[raw_feature_cols].copy()
        vg_features = vg_df[vg_feature_cols].copy()

        raw_features.columns = [f"raw_{c}" for c in raw_features.columns]
        vg_features.columns = [f"vg_{c}" for c in vg_features.columns]

        combined_df = pd.concat(
            [
                raw_df[meta_cols].reset_index(drop=True),
                raw_features.reset_index(drop=True),
                vg_features.reset_index(drop=True),
            ],
            axis=1
        )

        return combined_df

    raise ValueError("input_type must be 'raw', 'vg', or 'rawvg'")

def remove_baseline_neutral(df, emotion_col="emotion"):

    df = df.copy()

    original_count = len(df)

    df = df[
        ~df[emotion_col]
        .astype(str)
        .str.upper()
        .isin(["BASELINE", "NEUTRAL"])
    ]

    removed_count = original_count - len(df)

    print(
        f"Removed {removed_count} baseline/neutral samples. "
        f"Remaining samples: {len(df)}"
    )

    return df

def normalize_labels_subjectwise(df):

    df = df.copy()

    available_label_cols = [
        col for col in [VALENCE_COL, AROUSAL_COL]
        if col in df.columns
    ]

    if not available_label_cols:
        print(
            "Warning: No label columns found. "
            "Skipping subject-wise normalization."
        )
        return df

    for col in available_label_cols:

        df[col] = (
            df.groupby(SUBJECT_COL)[col]
            .transform(
                lambda x: (
                    (x - x.mean()) / x.std()
                ) if x.std() > 0 else 0
            )
        )

    print(
        f"Subject-wise Z-score applied to: "
        f"{available_label_cols}"
    )

    return df

def aggregate_features(df, prefix):
    df = df.copy()

    drop_columns = GROUP_COLS + WINDOW_COLUMNS + ALL_LABEL_COLS
    drop_columns = [col for col in drop_columns if col in df.columns]

    feature_cols = [
        col for col in df.columns
        if col not in drop_columns
    ]

    feature_cols = (
        df[feature_cols]
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    agg_df = (
        df.groupby(GROUP_COLS)[feature_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    agg_df.columns = [
        col if col in GROUP_COLS
        else f"{prefix}_{col}"
        for col in [
            "_".join(filter(None, map(str, col))).rstrip("_")
            if isinstance(col, tuple)
            else col
            for col in agg_df.columns
        ]
    ]

    return agg_df


def prepare_inputs(df, input_mode, classification):
    if classification not in ["binary", "ternary", "regression"]:
        raise ValueError("Classification not supported")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Required label column not found: {LABEL_COL}")

    if input_mode not in ["windowed", "aggregated"]:
        raise ValueError("input_mode must be either 'windowed' or 'aggregated'")

    if input_mode == "aggregated":
        df = aggregate_features(df, prefix="agg")

    drop_columns = [SUBJECT_COL, EMOTION_COL] + ALL_LABEL_COLS + WINDOW_COLUMNS
    drop_columns = [col for col in drop_columns if col in df.columns]
    
    X = df.drop(columns=drop_columns)

    X = X.select_dtypes(include=[np.number])
    X = X.astype(np.float32)

    labels = df[LABEL_COL].astype(float)

    if classification == "binary":
        if NORMALIZE_LABELS:
            y = (df[LABEL_COL].astype(float) > 0).astype(int)
        else:
            y = (df[LABEL_COL].astype(float) > 5).astype(int)
    if classification == "ternary":
        y = pd.Series(index=df.index, dtype=int)

        if NORMALIZE_LABELS == True:            
            y[labels < -1] = 0
            y[(labels >= -1) & (labels <= 1)] = 1
            y[labels > 1] = 2
        else:
            y[(labels >= 1) & (labels <= 3)] = 0
            y[(labels >= 4) & (labels <= 6)] = 1
            y[(labels >= 7) & (labels <= 9)] = 2
    if classification == "regression":
        y = labels

    groups = df[SUBJECT_COL]

    return X, y, groups


def get_models(fast=False):

    if fast:
        svm_grid = {
            "clf__kernel": ["rbf"],
            "clf__C": [5],
            "clf__gamma": ["scale"]
        }

        rf_grid = {
            "clf__n_estimators": [100],
            "clf__max_depth": [10],
            "clf__min_samples_leaf": [1],
            "clf__max_features": ["sqrt"]
        }

        knn_grid = {
            "clf__n_neighbors": [5],
            "clf__weights": ["distance"],
            "clf__p": [2]
        }

        xgb_grid = {
            "clf__n_estimators": [100],
            "clf__learning_rate": [0.1],
            "clf__max_depth": [3],
            "clf__min_child_weight": [1],
            "clf__subsample": [0.85],
            "clf__colsample_bytree": [0.85],
            "clf__reg_alpha": [0],
            "clf__reg_lambda": [1]
        }

    else:
        svm_grid = {
            "clf__kernel": ["rbf"],
            "clf__C": [1, 5, 10],
            "clf__gamma": ["scale"]
        }

        rf_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt"]
        }

        knn_grid = {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        }

        xgb_grid = {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
            "clf__min_child_weight": [1, 3],
            "clf__subsample": [0.85],
            "clf__colsample_bytree": [0.85],
            "clf__reg_alpha": [0, 0.01],
            "clf__reg_lambda": [1, 2]
        }

    # -------------------------
    # Model selection
    # -------------------------

    if CLASSIFICATION == "regression":

        svm_model = SVR()

        rf_model = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        knn_model = KNeighborsRegressor()

        xgb_model = xgb.XGBRegressor(
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            device="cuda",
            n_jobs=N_JOBS_XGB,
            verbosity=0
        )

    else:

        svm_model = SVC(
            random_state=RANDOM_STATE
        )

        rf_model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

        knn_model = KNeighborsClassifier()

        if CLASSIFICATION == "binary":

            xgb_model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                n_jobs=N_JOBS_XGB,
                verbosity=0
            )

        elif CLASSIFICATION == "ternary":

            xgb_model = xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                device="cuda",
                n_jobs=N_JOBS_XGB,
                verbosity=0
            )

        else:
            raise ValueError(
                "CLASSIFICATION must be binary, ternary or regression"
            )

    return {

        "SVM": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", svm_model)
            ]),
            "param_grid": svm_grid
        },

        "Random Forest": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("clf", rf_model)
            ]),
            "param_grid": rf_grid
        },

        "KNN": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                ("clf", knn_model)
            ]),
            "param_grid": knn_grid
        },

        "XGBoost": {
            "pipeline": Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("clf", xgb_model)
            ]),
            "param_grid": xgb_grid
        }
    }

def main(input_type="raw", input_mode="windowed", classification="binary"):
    df = load_dataset(input_type)

    if REMOVE_BASELINE_NEUTRAL:
        df = remove_baseline_neutral(df, emotion_col=EMOTION_COL)

    if NORMALIZE_LABELS:
        df = normalize_labels_subjectwise(df)

    X, y, groups = prepare_inputs(df, input_mode, classification)
    
    print(f"Number of features: {len(X.columns)}")
    print(f"\nTarget distribution ({classification}):")
    print(y.value_counts().sort_index())
    print(y.value_counts(normalize=True).sort_index())

    scoring_metric = SCORING
    
    if CLASSIFICATION == "regression":
        scoring_metric = "neg_mean_absolute_error"

    logo = LeaveOneGroupOut()
    models = get_models(FAST_MODE)

    results_all = {}
    subjects_results_all = []

    for model_name, model_config in models.items():
        print(f"\n🚀 Running {model_name} | Mode: {input_mode} ...")

        total_combinations = len(list(ParameterGrid(model_config["param_grid"])))
        print(f"  Number of grid combinations: {total_combinations}")

        fold_results = []
        subjects_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            subject_id = groups.iloc[test_idx].iloc[0]

            print(f"\n  Starting fold {fold_idx + 1} | Subject {subject_id}")

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            groups_train = groups.iloc[train_idx].to_numpy()

            n_unique_train_groups = len(np.unique(groups_train))
            n_inner_splits = min(INNER_CV_SPLITS, n_unique_train_groups)

            if n_inner_splits < 2:
                raise ValueError("Not enough training groups for inner GroupKFold.")

            inner_cv = GroupKFold(n_splits=n_inner_splits)

            search = GridSearchCV(
                estimator=model_config["pipeline"],
                param_grid=model_config["param_grid"],
                scoring=scoring_metric,
                cv=inner_cv,
                n_jobs=N_JOBS_SEARCH,
                verbose=1,
                pre_dispatch="1*n_jobs",
                refit=True
            )

            search.fit(X_train, y_train, groups=groups_train)

            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)


            if classification == "regression":
                metrics = {
                    "mae": mean_absolute_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "r2": r2_score(y_test, y_pred)
                }
            else:
                metrics = {
                    "acc": accuracy_score(y_test, y_pred),
                    "f1": f1_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0
                    ),
                    "precision": precision_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0
                    ),
                    "recall": recall_score(
                        y_test,
                        y_pred,
                        average="macro",
                        zero_division=0
                    )
                }

            fold_results.append(metrics)

            subjects_metrics.append({
                "model": model_name,
                "subject_id": subject_id,
                "threshold_used": 5,
                "neutral_label_removed": True,
                "best_params": search.best_params_,
                **metrics
            })


            if classification == "regression":
                print(
                    f"Completed fold {fold_idx + 1} | Subject {subject_id} | "
                    f"MAE={metrics['mae']:.4f} | "
                    f"RMSE={metrics['rmse']:.4f} | "
                    f"R2={metrics['r2']:.4f}"
                )
            else:
                print(
                    f"Completed fold {fold_idx + 1} | Subject {subject_id} | "
                    f"F1={metrics['f1']:.4f} | "
                    f"Acc={metrics['acc']:.4f}"
                )

        results_all[model_name] = fold_results
        subjects_results_all.append(pd.DataFrame(subjects_metrics))

    summary = []

    for model_name, res in results_all.items():
        if classification == "regression":
            summary.append({
                "Model": model_name,
                "MAE_mean": np.mean([r["mae"] for r in res]),
                "MAE_std": np.std([r["mae"] for r in res]),
                "RMSE_mean": np.mean([r["rmse"] for r in res]),
                "RMSE_std": np.std([r["rmse"] for r in res]),
                "R2_mean": np.mean([r["r2"] for r in res]),
                "R2_std": np.std([r["r2"] for r in res]),
            })
        else:
            summary.append({
                "Model": model_name,
                "Acc_mean": np.mean([r["acc"] for r in res]),
                "Acc_std": np.std([r["acc"] for r in res]),
                "F1_mean": np.mean([r["f1"] for r in res]),
                "F1_std": np.std([r["f1"] for r in res]),
                "Prec_mean": np.mean([r["precision"] for r in res]),
                "Prec_std": np.std([r["precision"] for r in res]),
                "Rec_mean": np.mean([r["recall"] for r in res]),
                "Rec_std": np.std([r["recall"] for r in res]),
            })

    summary_df = pd.DataFrame(summary)
    if classification != "regression":
        summary_df = summary_df.sort_values(by="F1_mean", ascending=False)

    print("\n===== FINAL RESULTS LOSO + GroupKFold inner CV + GridSearchCV =====\n")
    print(summary_df)

    norm_suffix = "normalized" if NORMALIZE_LABELS else "notNormalized"
    filter_suffix = "noBaselineNeutral" if REMOVE_BASELINE_NEUTRAL else "noFilter"
    fast_suffix = "fast" if FAST_MODE else "gridsearch"

    output_summary_path = OUTPUT_SUMMARY_PATH.format(
        target=TARGET_LABEL,
        input_type=input_type,
        mode=input_mode,
        class_name=classification,
        norm=norm_suffix,
        filter=filter_suffix,
        is_fast=fast_suffix
    )

    output_subjects_path = OUTPUT_SUBJECTS_PATH.format(
        target=TARGET_LABEL,
        input_type=input_type,
        mode=input_mode,
        class_name=classification,
        norm=norm_suffix,
        filter=filter_suffix,
        is_fast=fast_suffix
    )

    summary_df.to_csv(output_summary_path, index=False)
    print(f"\n✅ Saved: {output_summary_path}")

    subjects_results_df = pd.concat(subjects_results_all, ignore_index=True)
    subjects_results_df.to_csv(output_subjects_path, index=False)
    print(f"✅ Saved: {output_subjects_path}")


def run_all_modes():
    for feature_mode in FEATURE_MODE:
        for input_mode in INPUT_MODE:
            print("\n" + "=" * 80)
            print(f"Running FEATURE_MODE={feature_mode} | INPUT_MODE={input_mode}")
            print("=" * 80)

            main(input_type=feature_mode, input_mode=input_mode, classification=CLASSIFICATION)


if __name__ == "__main__":
    run_all_modes()