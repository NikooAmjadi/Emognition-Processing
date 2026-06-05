import h5py
import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from scipy.stats import kurtosis, skew

FILE_PATH = "emognition_complete.h5"
OUTPUT_PATH = "emognition_final_dataset_windowed.csv"

SUBJECTS = range(22, 65)

EMOTIONS = [
    "BASELINE", "ANGER", "ENTHUSIASM", "LIKING", "FEAR",
    "AMUSEMENT", "SADNESS", "NEUTRAL", "AWE", "DISGUST", "SURPRISE"
]

# اندازه هر پنجره به ثانیه
WINDOW_SIZE_SEC = 20
# مقدار همپوشانی هر پنجره به ثانیه
OVERLAP_SEC = 10

# نرخ نمونه برداری 
EDA_FS = 4
TEMP_FS = 4
BVP_FS = 64

# حداقل تعداد نمونه
MIN_EDA_LEN = 20
MIN_TEMP_LEN = 10
MIN_BVP_LEN = 64 * 20

USE_BASELINE_NORMALIZATION = True

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def safe_array(x):
    if x is None:
        return None
    return np.asarray(x).squeeze()


def safe_entropy(x):
    try:
        x = safe_array(x)
        if x is None or len(x) < 10:
            return np.nan
        hist, _ = np.histogram(x, bins=10, density=True)
        hist = hist[hist > 0]
        p = hist / np.sum(hist)
        return float(-np.sum(p * np.log2(p)))
    except:
        return np.nan


def compute_slope(x):
    x = safe_array(x)
    if x is None or len(x) < 2:
        return np.nan
    t = np.arange(len(x))
    return float(np.polyfit(t, x, 1)[0])


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


def extract_skt(temp, baseline=None):
    temp = safe_array(temp)
    if temp is None or len(temp) < MIN_TEMP_LEN:
        return None

    try:
        temp = nk.signal_detrend(temp)
    except:
        pass

    scale = 100.0

    feats = {
        "mean": float(np.mean(temp) * scale),
        "std": float(np.std(temp) * scale),
        "min": float(np.min(temp) * scale),
        "max": float(np.max(temp) * scale),
        "median": float(np.median(temp) * scale),
        "range": float((np.max(temp) - np.min(temp)) * scale),
        "entropy": safe_entropy(temp),
        "kurtosis": float(kurtosis(temp, nan_policy="omit")),
        "skewness": float(skew(temp, nan_policy="omit")),
        "slope": float(compute_slope(temp) * scale),
        "deriv_mean": float(np.mean(np.diff(temp)) * scale),
        "deriv_std": float(np.std(np.diff(temp)) * scale),
    }

    if USE_BASELINE_NORMALIZATION and baseline is not None:
        baseline = safe_array(baseline)
        if baseline is not None and len(baseline) > MIN_TEMP_LEN:
            eps = 1e-8
            feats["baseline_norm"] = float(
                np.log(np.mean(temp) + eps) - np.log(np.mean(baseline) + eps)
            )
        else:
            feats["baseline_norm"] = np.nan
    else:
        feats["baseline_norm"] = np.nan

    return feats


def extract_eda(eda):
    eda = safe_array(eda)
    if eda is None or len(eda) < MIN_EDA_LEN:
        return None

    try:
        eda_clean = nk.standardize(eda)

        signals, info = nk.eda_process(
            eda_clean,
            sampling_rate=EDA_FS,
            method="neurokit"
        )

        phasic = signals["EDA_Phasic"]
        scr_peaks = info.get("SCR_Peaks", [])

        feats = {
            "mean": float(np.mean(eda_clean)),
            "std": float(np.std(eda_clean)),
            "min": float(np.min(eda_clean)),
            "max": float(np.max(eda_clean)),
            "median": float(np.median(eda_clean)),
            "range": float(np.max(eda_clean) - np.min(eda_clean)),
            "entropy": safe_entropy(eda_clean),
            "kurtosis": float(kurtosis(eda_clean, nan_policy="omit")),
            "skewness": float(skew(eda_clean, nan_policy="omit")),
            "scr_count": int(len(scr_peaks)),
            "scr_amplitude": float(np.mean(phasic.iloc[scr_peaks])) if len(scr_peaks) > 0 else 0.0,
        }

        return feats

    except:
        feats = {
            "mean": float(np.mean(eda)),
            "std": float(np.std(eda)),
            "min": float(np.min(eda)),
            "max": float(np.max(eda)),
            "median": float(np.median(eda)),
            "range": float(np.max(eda) - np.min(eda)),
            "entropy": safe_entropy(eda),
            "kurtosis": float(kurtosis(eda, nan_policy="omit")),
            "skewness": float(skew(eda, nan_policy="omit")),
            "scr_count": np.nan,
            "scr_amplitude": np.nan,
        }
        return feats


def extract_ppg(ppg):
    ppg = safe_array(ppg)
    if ppg is None or len(ppg) < MIN_BVP_LEN:
        return None

    try:
        ppg_clean = nk.ppg_clean(ppg, sampling_rate=BVP_FS)
        signals, info = nk.ppg_process(ppg_clean, sampling_rate=BVP_FS)

        feats = {
            "mean": float(np.mean(ppg_clean)),
            "std": float(np.std(ppg_clean)),
            "min": float(np.min(ppg_clean)),
            "max": float(np.max(ppg_clean)),
            "range": float(np.max(ppg_clean) - np.min(ppg_clean)),
            "kurtosis": float(kurtosis(ppg_clean, nan_policy="omit")),
            "skewness": float(skew(ppg_clean, nan_policy="omit")),
        }

        try:
            hrv = nk.hrv_time(info, sampling_rate=BVP_FS, show=False)

            def safe(col):
                return float(hrv[col].values[0]) if col in hrv.columns else np.nan

            feats.update({
                "meanNN": safe("HRV_MeanNN"),
                "sdnn": safe("HRV_SDNN"),
                "rmssd": safe("HRV_RMSSD"),
                "pnn50": safe("HRV_pNN50"),
            })

        except:
            feats.update({
                "meanNN": np.nan,
                "sdnn": np.nan,
                "rmssd": np.nan,
                "pnn50": np.nan,
            })

        return feats

    except:
        return None


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


rows = []
error_log = []

total_subjects = len(list(SUBJECTS))

with h5py.File(FILE_PATH, "r") as f:

    for subject_index, subject_id in enumerate(SUBJECTS, start=1):

        progress = (subject_index / total_subjects) * 100
        print(f"Processing subject {subject_id} | {subject_index}/{total_subjects} | {progress:.1f}%")

        base_temp = load_values(
            f,
            f"/{subject_id}/devices/EMPATICA/BASELINE/STIMULUS/TEMP/values"
        )

        for emotion in EMOTIONS:

            path = f"/{subject_id}/devices/EMPATICA/{emotion}/STIMULUS"

            bvp = load_values(f, f"{path}/BVP/values")
            eda = load_values(f, f"{path}/EDA/values")
            temp = load_values(f, f"{path}/TEMP/values")

            if bvp is None and eda is None and temp is None:
                continue

            duration_sec = min(
                d for d in [
                    get_duration(bvp, BVP_FS),
                    get_duration(eda, EDA_FS),
                    get_duration(temp, TEMP_FS)
                ] if d > 0
            )

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

                    skt = extract_skt(temp_w, base_temp)
                    if skt is not None:
                        for k, v in skt.items():
                            row[f"SKT_{k}"] = v

                    eda_f = extract_eda(eda_w)
                    if eda_f is not None:
                        for k, v in eda_f.items():
                            row[f"EDA_{k}"] = v

                    ppg_f = extract_ppg(bvp_w)
                    if ppg_f is not None:
                        for k, v in ppg_f.items():
                            row[f"PPG_{k}"] = v

                    if len(row) > 8:
                        rows.append(row)

                except Exception as e:
                    error_log.append((subject_id, emotion, window_id, str(e)))


df = pd.DataFrame(rows)

df = df.dropna(axis=1, how="all")

df.to_csv(OUTPUT_PATH, index=False)

print("DONE ✔ Windowed dataset created successfully")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Errors:", len(error_log))

if len(error_log) > 0:
    print("\nSample errors:")
    for e in error_log[:5]:
        print(e)