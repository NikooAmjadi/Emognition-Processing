import h5py
import numpy as np
import pandas as pd
import networkx as nx
import warnings

FILE_PATH = "emognition_complete.h5"
OUTPUT_PATH = "emognition_vg_features.csv"

SUBJECTS = range(22, 65)

EMOTIONS = [
    "BASELINE", "ANGER", "ENTHUSIASM", "LIKING", "FEAR",
    "AMUSEMENT", "SADNESS", "NEUTRAL", "AWE", "DISGUST", "SURPRISE"
]

WINDOW_SIZE_SEC = 20
OVERLAP_SEC = 10

EDA_FS = 4
TEMP_FS = 4
BVP_FS = 64

MIN_EDA_LEN = 20
MIN_TEMP_LEN = 10
MIN_BVP_LEN = 64 * 20

MAX_BVP_POINTS = 300

warnings.filterwarnings("ignore")


def safe_array(x):
    if x is None:
        return None
    return np.asarray(x).squeeze()


def load_values(f, path):
    try:
        return np.array(f[path][:])
    except:
        return None


def load_label(f, subject_id, emotion):
    paths = [
        f"/{subject_id}/questionnaires/{emotion}/sam/VALENCE",
        f"/{subject_id}/questionnaires/{emotion}/SAM/VALENCE",
    ]
    for p in paths:
        try:
            return float(f[p][()])
        except:
            continue
    return np.nan


def get_duration(signal, fs):
    signal = safe_array(signal)
    if signal is None:
        return 0
    return len(signal) / fs


def get_window(signal, fs, start_sec, end_sec):
    signal = safe_array(signal)
    if signal is None:
        return None
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    if start_idx >= len(signal):
        return None
    return signal[start_idx:min(end_idx, len(signal))]


def create_windows(duration_sec):
    windows = []
    step_sec = WINDOW_SIZE_SEC - OVERLAP_SEC

    if step_sec <= 0:
        raise ValueError("OVERLAP_SEC must be smaller than WINDOW_SIZE_SEC")

    if duration_sec < WINDOW_SIZE_SEC:
        return windows

    start = 0
    window_id = 0

    while start + WINDOW_SIZE_SEC <= duration_sec:
        end = start + WINDOW_SIZE_SEC
        windows.append((window_id, start, end))
        start += step_sec
        window_id += 1

    return windows


def downsample_signal(x, max_points):
    x = safe_array(x)
    if x is None:
        return None

    if len(x) <= max_points:
        return x

    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return x[idx]


def natural_visibility_graph(x):
    x = safe_array(x)
    n = len(x)

    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            visible = True

            for k in range(i + 1, j):
                y_line = x[j] + (x[i] - x[j]) * ((j - k) / (j - i))
                if x[k] >= y_line:
                    visible = False
                    break

            if visible:
                graph.add_edge(i, j)

    return graph


def graph_entropy(values):
    try:
        values = np.asarray(values, dtype=float)
        values = values[values > 0]
        if len(values) == 0:
            return np.nan
        p = values / np.sum(values)
        return float(-np.sum(p * np.log2(p)))
    except:
        return np.nan


def extract_vg_features(signal, min_len, prefix, max_points=None):
    signal = safe_array(signal)

    if signal is None or len(signal) < min_len:
        return None

    if max_points is not None:
        signal = downsample_signal(signal, max_points)

    if signal is None or len(signal) < 5:
        return None

    signal = signal.astype(float)

    if np.all(np.isnan(signal)):
        return None

    signal = np.nan_to_num(signal, nan=np.nanmean(signal))

    if np.std(signal) > 0:
        signal = (signal - np.mean(signal)) / np.std(signal)

    graph = natural_visibility_graph(signal)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return None

    degrees = np.array([d for _, d in graph.degree()], dtype=float)

    feats = {
        f"{prefix}_vg_nodes": float(n_nodes),
        f"{prefix}_vg_edges": float(n_edges),
        f"{prefix}_vg_density": float(nx.density(graph)),
        f"{prefix}_vg_degree_mean": float(np.mean(degrees)),
        f"{prefix}_vg_degree_std": float(np.std(degrees)),
        f"{prefix}_vg_degree_min": float(np.min(degrees)),
        f"{prefix}_vg_degree_max": float(np.max(degrees)),
        f"{prefix}_vg_degree_median": float(np.median(degrees)),
        f"{prefix}_vg_degree_entropy": graph_entropy(degrees),
        f"{prefix}_vg_clustering_mean": float(np.mean(list(nx.clustering(graph).values()))),
        f"{prefix}_vg_transitivity": float(nx.transitivity(graph)),
    }

    try:
        if nx.is_connected(graph):
            feats[f"{prefix}_vg_avg_path_length"] = float(nx.average_shortest_path_length(graph))
            feats[f"{prefix}_vg_diameter"] = float(nx.diameter(graph))
        else:
            largest_cc = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
            feats[f"{prefix}_vg_avg_path_length"] = float(nx.average_shortest_path_length(largest_cc))
            feats[f"{prefix}_vg_diameter"] = float(nx.diameter(largest_cc))
    except:
        feats[f"{prefix}_vg_avg_path_length"] = np.nan
        feats[f"{prefix}_vg_diameter"] = np.nan

    try:
        betweenness = nx.betweenness_centrality(graph, normalized=True)
        feats[f"{prefix}_vg_betweenness_mean"] = float(np.mean(list(betweenness.values())))
        feats[f"{prefix}_vg_betweenness_std"] = float(np.std(list(betweenness.values())))
    except:
        feats[f"{prefix}_vg_betweenness_mean"] = np.nan
        feats[f"{prefix}_vg_betweenness_std"] = np.nan

    try:
        closeness = nx.closeness_centrality(graph)
        feats[f"{prefix}_vg_closeness_mean"] = float(np.mean(list(closeness.values())))
        feats[f"{prefix}_vg_closeness_std"] = float(np.std(list(closeness.values())))
    except:
        feats[f"{prefix}_vg_closeness_mean"] = np.nan
        feats[f"{prefix}_vg_closeness_std"] = np.nan

    return feats


rows = []
error_log = []

total_subjects = len(list(SUBJECTS))

with h5py.File(FILE_PATH, "r") as f:

    for subject_index, subject_id in enumerate(SUBJECTS, start=1):

        progress = (subject_index / total_subjects) * 100
        print(f"Processing subject {subject_id} | {subject_index}/{total_subjects} | {progress:.1f}%")

        for emotion in EMOTIONS:

            path = f"/{subject_id}/devices/EMPATICA/{emotion}/STIMULUS"

            bvp = load_values(f, f"{path}/BVP/values")
            eda = load_values(f, f"{path}/EDA/values")
            temp = load_values(f, f"{path}/TEMP/values")

            if bvp is None and eda is None and temp is None:
                continue

            durations = [
                get_duration(bvp, BVP_FS),
                get_duration(eda, EDA_FS),
                get_duration(temp, TEMP_FS)
            ]

            durations = [d for d in durations if d > 0]

            if len(durations) == 0:
                continue

            duration_sec = min(durations)

            windows = create_windows(duration_sec)

            for window_id, start_sec, end_sec in windows:

                try:
                    row = {
                        "subject_id": subject_id,
                        "emotion": emotion,
                        "window_id": window_id,
                        "window_start_sec": start_sec,
                        "window_end_sec": end_sec,
                        "window_size_sec": WINDOW_SIZE_SEC,
                        "overlap_sec": OVERLAP_SEC,
                        "label_valence": load_label(f, subject_id, emotion)
                    }

                    temp_w = get_window(temp, TEMP_FS, start_sec, end_sec)
                    eda_w = get_window(eda, EDA_FS, start_sec, end_sec)
                    bvp_w = get_window(bvp, BVP_FS, start_sec, end_sec)

                    temp_vg = extract_vg_features(
                        temp_w,
                        MIN_TEMP_LEN,
                        "TEMP",
                        max_points=None
                    )

                    eda_vg = extract_vg_features(
                        eda_w,
                        MIN_EDA_LEN,
                        "EDA",
                        max_points=None
                    )

                    bvp_vg = extract_vg_features(
                        bvp_w,
                        MIN_BVP_LEN,
                        "BVP",
                        max_points=MAX_BVP_POINTS
                    )

                    if temp_vg is not None:
                        row.update(temp_vg)

                    if eda_vg is not None:
                        row.update(eda_vg)

                    if bvp_vg is not None:
                        row.update(bvp_vg)

                    if len(row) > 8:
                        rows.append(row)

                except Exception as e:
                    error_log.append((subject_id, emotion, window_id, str(e)))


df = pd.DataFrame(rows)

df = df.dropna(axis=1, how="all")

df.to_csv(OUTPUT_PATH, index=False)

print("DONE ✔ Visibility Graph dataset created successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Errors:", len(error_log))

if len(error_log) > 0:
    print("\nSample errors:")
    for e in error_log[:5]:
        print(e)