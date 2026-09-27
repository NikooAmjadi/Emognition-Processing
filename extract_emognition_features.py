#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "h5py==3.14.0",
#   "networkx==3.4.2",
#   "neurokit2==0.2.13",
#   "numpy==2.2.6",
#   "pandas==2.2.3",
#   "scipy==1.15.3",
#   "ts2vg==1.2.4",
# ]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

try:
    import h5py
    import networkx as nx
    import neurokit2 as nk
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import signal as scipy_signal
    from scipy import stats
    from ts2vg import HorizontalVG, NaturalVG
    import ts2vg
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing a required package. Use Python 3.11 and install exactly:\n"
        "  python -m pip install h5py==3.14.0 networkx==3.4.2 "
        "neurokit2==0.2.13 numpy==2.2.6 pandas==2.2.3 scipy==1.15.3 "
        "ts2vg==1.2.4\n"
        f"Original import error: {exc}"
    ) from exc


GENERATOR_NAME = "extract_emognition_features.py"
GENERATOR_VERSION = "1.1.0"
EXPECTED_SCHEMA = "emognition_empatica_sam"
SUPPORTED_SCHEMA_MAJOR = 1
REQUIRED_SIGNALS = ("BVP", "EDA", "TEMP")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "emognition_empatica_sam.h5"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "features"
EPSILON = np.finfo(float).eps
EXPECTED_FEATURE_COUNTS = {"handcrafted": 29, "nvg": 48, "hvg": 48}

METADATA_COLUMNS = [
    "subject_id",
    "emotion",
    "movie_order_index",
    "window_id",
    "window_size_sec",
    "overlap_sec",
    "window_start_sec",
    "window_end_sec",
    "common_duration_sec",
    "sam_valence",
    "sam_arousal",
    "sam_motivation",
]

QC_COLUMNS = [
    "qc_bvp_samples",
    "qc_eda_samples",
    "qc_temp_samples",
    "qc_vg_points",
]


class ExtractionError(RuntimeError):
    pass


@dataclass(frozen=True)
class WindowSetting:
    window_sec: float
    overlap_sec: float

    @property
    def step_sec(self) -> float:
        return self.window_sec - self.overlap_sec

    @staticmethod
    def _token(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    @property
    def slug(self) -> str:
        return f"w{self._token(self.window_sec)}_o{self._token(self.overlap_sec)}"


@dataclass
class SignalData:
    timestamps: np.ndarray
    raw: np.ndarray
    processed: np.ndarray
    sampling_rate_hz: float


@dataclass
class TrialData:
    bvp: SignalData
    eda: SignalData
    temp: SignalData
    scr_peaks: np.ndarray
    scr_amplitudes: np.ndarray
    ppg_peaks: np.ndarray
    common_start_epoch_s: float
    common_end_epoch_s: float

    @property
    def common_duration_sec(self) -> float:
        return self.common_end_epoch_s - self.common_start_epoch_s


class Audit:
    def __init__(self) -> None:
        self.warning_counts: Counter[str] = Counter()
        self.skipped_trials: list[dict[str, str]] = []
        self.no_window_trials: dict[str, list[str]] = defaultdict(list)
        self.graph_seconds: Counter[str] = Counter()
        self.graph_count: Counter[str] = Counter()

    def warn(self, category: str, amount: int = 1) -> None:
        if amount > 0:
            self.warning_counts[category] += amount

    def skip(self, subject: str, emotion: str, reason: str) -> None:
        self.skipped_trials.append(
            {"subject_id": subject, "emotion": emotion, "reason": reason}
        )


def parse_setting(text: str) -> WindowSetting:
    try:
        window_text, overlap_text = text.split(":", maxsplit=1)
        setting = WindowSetting(float(window_text), float(overlap_text))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid setting {text!r}; expected WINDOW:OVERLAP, for example 60:30"
        ) from exc
    if not math.isfinite(setting.window_sec) or setting.window_sec <= 0:
        raise argparse.ArgumentTypeError("Window length must be a positive finite number")
    if not math.isfinite(setting.overlap_sec) or setting.overlap_sec < 0:
        raise argparse.ArgumentTypeError("Overlap must be a non-negative finite number")
    if setting.overlap_sec >= setting.window_sec:
        raise argparse.ArgumentTypeError("Overlap must be smaller than the window")
    return setting


def parse_emotion_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result = {
        item.strip().upper()
        for value in values
        for item in value.split(",")
        if item.strip()
    }
    return result or None


def parse_identifier_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result = {
        item.strip()
        for value in values
        for item in value.split(",")
        if item.strip()
    }
    return result or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create aligned handcrafted, Natural Visibility Graph, and "
            "Horizontal Visibility Graph features from the reproducible HDF5."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-hdf5", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="emognition_features")
    parser.add_argument(
        "--setting",
        action="append",
        type=parse_setting,
        help="Repeatable WINDOW:OVERLAP pair in seconds; default is 60:30.",
    )
    parser.add_argument(
        "--vg-sampling-rate",
        type=float,
        default=4.0,
        help="Common anti-aliased sampling rate used by both graph families.",
    )
    parser.add_argument(
        "--include-emotions",
        action="append",
        help="Optional comma-separated emotion names; repeatable.",
    )
    parser.add_argument(
        "--exclude-emotions",
        action="append",
        help="Optional comma-separated emotion names; repeatable.",
    )
    parser.add_argument(
        "--include-participants",
        action="append",
        help="Optional comma-separated participant IDs; repeatable.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop instead of recording and skipping an unavailable/unusable trial.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing CSV/manifest files."
    )
    return parser


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def finite_interpolate(values: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=float).reshape(-1)
    bad = ~np.isfinite(array)
    bad_count = int(bad.sum())
    if not bad_count:
        return array, 0
    good_indices = np.flatnonzero(~bad)
    if not len(good_indices):
        raise ExtractionError("Signal contains no finite values")
    indices = np.arange(len(array))
    array[bad] = np.interp(indices[bad], good_indices, array[good_indices])
    return array, bad_count


def sample_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


def population_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=0)) if len(values) else math.nan


def histogram_entropy(values: np.ndarray, bins: int = 10) -> float:
    if len(values) < 2 or np.ptp(values) <= EPSILON:
        return 0.0
    counts, _ = np.histogram(values, bins=bins)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def safe_skew(values: np.ndarray) -> float:
    if len(values) < 3 or np.ptp(values) <= EPSILON:
        return 0.0
    return float(stats.skew(values, bias=True))


def safe_kurtosis(values: np.ndarray) -> float:
    if len(values) < 4 or np.ptp(values) <= EPSILON:
        return 0.0
    return float(stats.kurtosis(values, fisher=True, bias=True))


def get_sampling_rate(signal_group: h5py.Group, timestamps: np.ndarray) -> float:
    for name in ("nominal_sampling_rate_hz", "estimated_sampling_rate_hz"):
        value = float(signal_group.attrs.get(name, math.nan))
        if math.isfinite(value) and value > 0:
            return value
    if len(timestamps) > 1:
        delta = float(np.median(np.diff(timestamps)))
        if delta > 0:
            return 1.0 / delta
    raise ExtractionError(f"Cannot determine sampling rate for {signal_group.name}")


def read_raw_signal(signals: h5py.Group, name: str, audit: Audit) -> SignalData:
    if name not in signals:
        raise ExtractionError(f"Required signal {name} is absent")
    group = signals[name]
    timestamps = np.asarray(group["timestamps_epoch_s"][:], dtype=float)
    values = np.asarray(group["values"][:, 0], dtype=float)
    if len(timestamps) != len(values) or len(values) < 4:
        raise ExtractionError(f"Invalid {name} timestamp/value shape")
    if np.any(np.diff(timestamps) < 0):
        raise ExtractionError(f"{name} timestamps are not monotonic")
    values, bad_count = finite_interpolate(values)
    if bad_count:
        audit.warn(f"{name.lower()}_nonfinite_values_interpolated", bad_count)
    return SignalData(
        timestamps=timestamps,
        raw=values,
        processed=values.copy(),
        sampling_rate_hz=get_sampling_rate(group, timestamps),
    )


def recording_end(signal_data: SignalData) -> float:
    timestamps = signal_data.timestamps
    if len(timestamps) > 1:
        positive_deltas = np.diff(timestamps)
        positive_deltas = positive_deltas[positive_deltas > 0]
        delta = (
            float(np.median(positive_deltas))
            if len(positive_deltas)
            else 1.0 / signal_data.sampling_rate_hz
        )
    else:
        delta = 1.0 / signal_data.sampling_rate_hz
    return float(timestamps[-1] + delta)


def capture_warnings(category: str, audit: Audit):
    class WarningCapture:
        def __enter__(self):
            self.context = warnings.catch_warnings(record=True)
            self.records = self.context.__enter__()
            warnings.simplefilter("always")
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.context.__exit__(exc_type, exc, traceback)
            if self.records:
                audit.warn(category, len(self.records))
            return False

    return WarningCapture()


def process_trial(signals: h5py.Group, audit: Audit) -> TrialData:
    bvp = read_raw_signal(signals, "BVP", audit)
    eda = read_raw_signal(signals, "EDA", audit)
    temp = read_raw_signal(signals, "TEMP", audit)

    with capture_warnings("neurokit_ppg_warnings", audit):
        ppg_signals, _ppg_info = nk.ppg_process(
            bvp.raw, sampling_rate=bvp.sampling_rate_hz
        )
    bvp.processed, bad = finite_interpolate(ppg_signals["PPG_Clean"].to_numpy())
    audit.warn("ppg_clean_nonfinite_values_interpolated", bad)
    ppg_peaks = np.flatnonzero(np.asarray(ppg_signals["PPG_Peaks"], dtype=int) == 1)

    with capture_warnings("neurokit_eda_warnings", audit):
        eda_signals, _eda_info = nk.eda_process(
            eda.raw, sampling_rate=eda.sampling_rate_hz
        )
    eda.processed, bad = finite_interpolate(eda_signals["EDA_Clean"].to_numpy())
    audit.warn("eda_clean_nonfinite_values_interpolated", bad)
    scr_peaks = np.flatnonzero(np.asarray(eda_signals["SCR_Peaks"], dtype=int) == 1)
    scr_amplitudes = np.asarray(eda_signals["SCR_Amplitude"], dtype=float)[scr_peaks]

    temp.processed = scipy_signal.detrend(temp.raw, type="linear")
    common_start = max(
        bvp.timestamps[0], eda.timestamps[0], temp.timestamps[0]
    )
    common_end = min(recording_end(bvp), recording_end(eda), recording_end(temp))
    if common_end <= common_start:
        raise ExtractionError("Signals have no overlapping time interval")

    return TrialData(
        bvp=bvp,
        eda=eda,
        temp=temp,
        scr_peaks=scr_peaks,
        scr_amplitudes=scr_amplitudes,
        ppg_peaks=ppg_peaks,
        common_start_epoch_s=float(common_start),
        common_end_epoch_s=float(common_end),
    )


def mask_for_window(timestamps: np.ndarray, start: float, end: float) -> np.ndarray:
    return (timestamps >= start) & (timestamps < end)


def hrv_features(
    peak_indices: np.ndarray,
    window_mask: np.ndarray,
    sampling_rate_hz: float,
    audit: Audit,
) -> dict[str, float]:
    names = {
        "HRV_MeanNN": "hc_ppg_meanNN",
        "HRV_SDNN": "hc_ppg_sdnn",
        "HRV_RMSSD": "hc_ppg_rmssd",
        "HRV_pNN50": "hc_ppg_pnn50",
    }
    result = {output: math.nan for output in names.values()}
    positions = np.flatnonzero(window_mask)
    if not len(positions):
        audit.warn("hrv_empty_windows")
        return result
    selected = peak_indices[(peak_indices >= positions[0]) & (peak_indices <= positions[-1])]
    local = selected - positions[0]
    if len(local) < 4:
        audit.warn("hrv_insufficient_peaks")
        return result
    try:
        with capture_warnings("neurokit_hrv_warnings", audit):
            table = nk.hrv_time(
                {"PPG_Peaks": local}, sampling_rate=sampling_rate_hz, show=False
            )
        for source, output in names.items():
            if source in table:
                result[output] = float(table[source].iloc[0])
    except (ValueError, TypeError, IndexError, ZeroDivisionError) as exc:
        audit.warn(f"hrv_failed_{type(exc).__name__}")
    return result


def handcrafted_features(
    trial: TrialData,
    masks: dict[str, np.ndarray],
    baseline_temp_mean: float,
    baseline_temp_std: float,
    audit: Audit,
) -> dict[str, float]:
    bvp_mask, eda_mask, temp_mask = masks["BVP"], masks["EDA"], masks["TEMP"]
    bvp_values = trial.bvp.processed[bvp_mask]
    eda_values = trial.eda.processed[eda_mask]
    temp_values = trial.temp.processed[temp_mask]
    raw_temp_values = trial.temp.raw[temp_mask]

    result: dict[str, float] = {
        "hc_skt_std": population_std(temp_values),
        "hc_skt_max": float(np.max(temp_values)),
        "hc_skt_median": float(np.median(temp_values)),
        "hc_skt_range": float(np.ptp(temp_values)),
        "hc_skt_entropy": histogram_entropy(temp_values),
        "hc_skt_kurtosis": safe_kurtosis(temp_values),
        "hc_skt_skewness": safe_skew(temp_values),
        "hc_eda_max": float(np.max(eda_values)),
        "hc_eda_median": float(np.median(eda_values)),
        "hc_eda_range": float(np.ptp(eda_values)),
        "hc_eda_entropy": histogram_entropy(eda_values),
        "hc_eda_kurtosis": safe_kurtosis(eda_values),
        "hc_eda_skewness": safe_skew(eda_values),
        "hc_ppg_mean": float(np.mean(bvp_values)),
        "hc_ppg_std": population_std(bvp_values),
        "hc_ppg_min": float(np.min(bvp_values)),
        "hc_ppg_max": float(np.max(bvp_values)),
        "hc_ppg_range": float(np.ptp(bvp_values)),
        "hc_ppg_kurtosis": safe_kurtosis(bvp_values),
        "hc_ppg_skewness": safe_skew(bvp_values),
    }

    eda_positions = np.flatnonzero(eda_mask)
    peak_selector = (
        (trial.scr_peaks >= eda_positions[0])
        & (trial.scr_peaks <= eda_positions[-1])
    )
    amplitudes = trial.scr_amplitudes[peak_selector]
    finite_amplitudes = amplitudes[np.isfinite(amplitudes)]
    scr_count = int(peak_selector.sum())
    result["hc_eda_scr_count"] = float(scr_count)
    result["hc_eda_scr_amplitude"] = (
        float(np.mean(finite_amplitudes)) if len(finite_amplitudes) else 0.0
    )

    temp_derivative = np.gradient(temp_values, trial.temp.timestamps[temp_mask])
    result["hc_skt_deriv_mean"] = float(np.mean(temp_derivative))
    result["hc_skt_deriv_std"] = population_std(temp_derivative)
    raw_temp_mean = float(np.mean(raw_temp_values))
    result["hc_skt_baseline_norm"] = (
        (raw_temp_mean - baseline_temp_mean) / baseline_temp_std
        if math.isfinite(baseline_temp_mean)
        and math.isfinite(baseline_temp_std)
        and baseline_temp_std > EPSILON
        else math.nan
    )
    result.update(
        hrv_features(
            trial.ppg_peaks, bvp_mask, trial.bvp.sampling_rate_hz, audit
        )
    )
    return result


def resample_for_graph(
    values: np.ndarray,
    source_rate_hz: float,
    target_rate_hz: float,
    target_points: int,
) -> np.ndarray:
    values, _ = finite_interpolate(values)
    ratio = Fraction(target_rate_hz / source_rate_hz).limit_denominator(1000)
    resampled = scipy_signal.resample_poly(values, ratio.numerator, ratio.denominator)
    if len(resampled) != target_points:
        resampled = scipy_signal.resample(resampled, target_points)
    resampled = np.asarray(resampled, dtype=float)
    scale = float(np.std(resampled))
    if scale > EPSILON:
        resampled = (resampled - np.mean(resampled)) / scale
    else:
        resampled = resampled - np.mean(resampled)
    return resampled


def graph_features(values: np.ndarray, family: str, prefix: str, audit: Audit) -> dict[str, float]:
    started = time.perf_counter()
    visibility_graph = NaturalVG() if family == "nvg" else HorizontalVG()
    visibility_graph.build(values)
    graph = visibility_graph.as_networkx()
    elapsed = time.perf_counter() - started
    audit.graph_seconds[family] += elapsed
    audit.graph_count[family] += 1

    n_edges = graph.number_of_edges()
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    degree_values, degree_counts = np.unique(degrees, return_counts=True)
    del degree_values
    probabilities = degree_counts / degree_counts.sum()
    degree_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    clustering = np.asarray(list(nx.clustering(graph).values()), dtype=float)

    if nx.is_connected(graph):
        connected_graph = graph
    else:
        nodes = max(nx.connected_components(graph), key=len)
        connected_graph = graph.subgraph(nodes).copy()

    betweenness = np.asarray(
        list(nx.betweenness_centrality(graph, normalized=True).values()), dtype=float
    )
    closeness = np.asarray(list(nx.closeness_centrality(graph).values()), dtype=float)
    root = f"{family}_{prefix}"
    return {
        f"{root}_edges": float(n_edges),
        f"{root}_density": float(nx.density(graph)),
        f"{root}_degree_mean": float(np.mean(degrees)),
        f"{root}_degree_std": population_std(degrees),
        f"{root}_degree_min": float(np.min(degrees)),
        f"{root}_degree_max": float(np.max(degrees)),
        f"{root}_degree_median": float(np.median(degrees)),
        f"{root}_degree_entropy": degree_entropy,
        f"{root}_clustering_mean": float(np.mean(clustering)),
        f"{root}_transitivity": float(nx.transitivity(graph)),
        f"{root}_avg_path_length": float(
            nx.average_shortest_path_length(connected_graph)
        ),
        f"{root}_diameter": float(nx.diameter(connected_graph)),
        f"{root}_betweenness_mean": float(np.mean(betweenness)),
        f"{root}_betweenness_std": population_std(betweenness),
        f"{root}_closeness_mean": float(np.mean(closeness)),
        f"{root}_closeness_std": population_std(closeness),
    }


def visibility_features(
    trial: TrialData,
    masks: dict[str, np.ndarray],
    target_rate_hz: float,
    target_points: int,
    audit: Audit,
) -> dict[str, float]:
    sources = {
        "bvp": (trial.bvp.processed[masks["BVP"]], trial.bvp.sampling_rate_hz),
        "eda": (trial.eda.processed[masks["EDA"]], trial.eda.sampling_rate_hz),
        "temp": (trial.temp.processed[masks["TEMP"]], trial.temp.sampling_rate_hz),
    }
    result: dict[str, float] = {}
    for modality, (values, source_rate) in sources.items():
        graph_values = resample_for_graph(
            values, source_rate, target_rate_hz, target_points
        )
        result.update(graph_features(graph_values, "nvg", modality, audit))
        result.update(graph_features(graph_values, "hvg", modality, audit))
    return result


def window_starts(duration_sec: float, setting: WindowSetting) -> Iterable[float]:
    index = 0
    tolerance = 1e-6
    while index * setting.step_sec + setting.window_sec <= duration_sec + tolerance:
        yield index * setting.step_sec
        index += 1


def temp_baseline_stats(participant_group: h5py.Group, audit: Audit) -> tuple[float, float]:
    path = "trials/BASELINE/phases/STIMULUS"
    if path not in participant_group:
        audit.warn("missing_temperature_baseline")
        return math.nan, math.nan
    phase = participant_group[path]
    if not bool(phase.attrs.get("available", False)) or "signals/TEMP" not in phase:
        audit.warn("missing_temperature_baseline")
        return math.nan, math.nan
    values = np.asarray(phase["signals/TEMP/values"][:, 0], dtype=float)
    try:
        values, bad = finite_interpolate(values)
    except ExtractionError:
        audit.warn("invalid_temperature_baseline")
        return math.nan, math.nan
    audit.warn("temperature_baseline_nonfinite_values_interpolated", bad)
    return float(np.mean(values)), sample_std(values)


def ordered_participants(participants: h5py.Group) -> list[str]:
    def key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    return sorted(participants.keys(), key=key)


def validate_input(h5: h5py.File) -> None:
    schema = str(h5.attrs.get("schema_name", ""))
    version = str(h5.attrs.get("schema_version", ""))
    if schema != EXPECTED_SCHEMA:
        raise ExtractionError(
            f"Unsupported HDF5 schema {schema!r}; expected {EXPECTED_SCHEMA!r}"
        )
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise ExtractionError(f"Invalid HDF5 schema version {version!r}") from exc
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise ExtractionError(f"Unsupported HDF5 schema version {version!r}")
    if "participants" not in h5:
        raise ExtractionError("HDF5 does not contain /participants")


def output_paths(args: argparse.Namespace, settings: list[WindowSetting]) -> dict[str, tuple[Path, Path]]:
    paths: dict[str, tuple[Path, Path]] = {}
    for setting in settings:
        stem = f"{args.output_prefix}_{setting.slug}"
        paths[setting.slug] = (
            args.output_dir / f"{stem}.csv",
            args.output_dir / f"{stem}.manifest.json",
        )
    return paths


def dependency_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "h5py": h5py.__version__,
        "networkx": nx.__version__,
        "neurokit2": nk.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "ts2vg": ts2vg.__version__,
    }


def atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".building")
    try:
        table.to_csv(temporary, index=False, float_format="%.12g", na_rep="NaN")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".building")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def print_startup(
    input_path: Path,
    output_dir: Path,
    settings: list[WindowSetting],
    graph_rate: float,
) -> None:
    print("=" * 72)
    print("Emognition reproducible feature extraction")
    print("Handcrafted features + Natural VG (NVG) + Horizontal VG (HVG)")
    print("Feature schema: 29 handcrafted + 48 NVG + 48 HVG = 125")
    print(f"Input:          {input_path}")
    print(f"Output folder:  {output_dir}")
    print("Settings:       " + ", ".join(f"{s.window_sec:g}s/{s.overlap_sec:g}s" for s in settings))
    print(f"VG sample rate: {graph_rate:g} Hz (anti-aliased, equal point count)")
    print("This step computes many graph metrics and may take a long time.")
    print("Progress will be printed once for every participant. Please wait.")
    print("=" * 72, flush=True)


def run(args: argparse.Namespace) -> int:
    input_path = args.input_hdf5.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise ExtractionError(f"Input HDF5 does not exist: {input_path}")
    if not math.isfinite(args.vg_sampling_rate) or args.vg_sampling_rate <= 0:
        raise ExtractionError("--vg-sampling-rate must be a positive finite number")

    settings = args.setting or [WindowSetting(60.0, 30.0)]
    unique_settings: list[WindowSetting] = []
    seen_settings: set[WindowSetting] = set()
    for setting in settings:
        if setting not in seen_settings:
            unique_settings.append(setting)
            seen_settings.add(setting)
    settings = unique_settings
    include = parse_emotion_filter(args.include_emotions)
    exclude = parse_emotion_filter(args.exclude_emotions) or set()
    included_participants = parse_identifier_filter(args.include_participants)
    if include and include & exclude:
        overlap = ", ".join(sorted(include & exclude))
        raise ExtractionError(f"Emotion(s) both included and excluded: {overlap}")

    paths = output_paths(args, settings)
    existing = [path for pair in paths.values() for path in pair if path.exists()]
    if existing and not args.overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise ExtractionError(
            "Output already exists; use --overwrite or choose another prefix:\n  " + joined
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    print_startup(input_path, output_dir, settings, args.vg_sampling_rate)

    audit = Audit()
    rows_by_setting: dict[str, list[dict[str, Any]]] = {
        setting.slug: [] for setting in settings
    }
    started = time.perf_counter()
    input_hash = sha256_file(input_path)

    with h5py.File(input_path, "r") as h5:
        validate_input(h5)
        participants = h5["participants"]
        participant_ids = ordered_participants(participants)
        if included_participants is not None:
            missing_participants = included_participants - set(participant_ids)
            if missing_participants:
                raise ExtractionError(
                    "Requested participant(s) absent from HDF5: "
                    + ", ".join(sorted(missing_participants))
                )
            participant_ids = [
                value for value in participant_ids if value in included_participants
            ]
        total_participants = len(participant_ids)
        for participant_number, subject_id in enumerate(participant_ids, start=1):
            participant_started = time.perf_counter()
            participant = participants[subject_id]
            baseline_mean, baseline_std = temp_baseline_stats(participant, audit)
            trials = participant["trials"]
            ordered_trials = sorted(
                trials.keys(), key=lambda name: int(trials[name].attrs["movie_order_index"])
            )
            participant_rows_before = sum(len(rows) for rows in rows_by_setting.values())
            for emotion in ordered_trials:
                emotion_upper = emotion.upper()
                if include is not None and emotion_upper not in include:
                    continue
                if emotion_upper in exclude:
                    continue
                trial_group = trials[emotion]
                phase_path = "phases/STIMULUS"
                if phase_path not in trial_group:
                    reason = "STIMULUS phase group is absent"
                    if args.strict:
                        raise ExtractionError(f"{subject_id}/{emotion}: {reason}")
                    audit.skip(subject_id, emotion, reason)
                    continue
                phase = trial_group[phase_path]
                if not bool(phase.attrs.get("available", False)):
                    reason = "STIMULUS source file is unavailable"
                    if args.strict:
                        raise ExtractionError(f"{subject_id}/{emotion}: {reason}")
                    audit.skip(subject_id, emotion, reason)
                    continue
                if "signals" not in phase:
                    reason = "STIMULUS signal group is absent"
                    if args.strict:
                        raise ExtractionError(f"{subject_id}/{emotion}: {reason}")
                    audit.skip(subject_id, emotion, reason)
                    continue
                try:
                    trial = process_trial(phase["signals"], audit)
                except (ExtractionError, ValueError, TypeError, IndexError) as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    if args.strict:
                        raise ExtractionError(f"{subject_id}/{emotion}: {reason}") from exc
                    audit.skip(subject_id, emotion, reason)
                    continue

                for setting in settings:
                    target_points = int(round(setting.window_sec * args.vg_sampling_rate))
                    if target_points < 5:
                        raise ExtractionError(
                            f"{setting.slug} produces only {target_points} VG points; "
                            "increase the window or --vg-sampling-rate"
                        )
                    starts = list(window_starts(trial.common_duration_sec, setting))
                    if not starts:
                        audit.no_window_trials[setting.slug].append(
                            f"{subject_id}/{emotion}"
                        )
                        continue
                    for window_id, relative_start in enumerate(starts):
                        start_epoch = trial.common_start_epoch_s + relative_start
                        end_epoch = start_epoch + setting.window_sec
                        masks = {
                            "BVP": mask_for_window(
                                trial.bvp.timestamps, start_epoch, end_epoch
                            ),
                            "EDA": mask_for_window(
                                trial.eda.timestamps, start_epoch, end_epoch
                            ),
                            "TEMP": mask_for_window(
                                trial.temp.timestamps, start_epoch, end_epoch
                            ),
                        }
                        if any(int(mask.sum()) < 4 for mask in masks.values()):
                            audit.warn("windows_with_insufficient_signal_samples")
                            continue
                        row: dict[str, Any] = {
                            "subject_id": int(subject_id) if subject_id.isdigit() else subject_id,
                            "emotion": emotion,
                            "movie_order_index": int(trial_group.attrs["movie_order_index"]),
                            "window_id": window_id,
                            "window_size_sec": setting.window_sec,
                            "overlap_sec": setting.overlap_sec,
                            "window_start_sec": relative_start,
                            "window_end_sec": relative_start + setting.window_sec,
                            "common_duration_sec": trial.common_duration_sec,
                            "sam_valence": int(trial_group.attrs["sam_valence"]),
                            "sam_arousal": int(trial_group.attrs["sam_arousal"]),
                            "sam_motivation": int(trial_group.attrs["sam_motivation"]),
                            "qc_bvp_samples": int(masks["BVP"].sum()),
                            "qc_eda_samples": int(masks["EDA"].sum()),
                            "qc_temp_samples": int(masks["TEMP"].sum()),
                            "qc_vg_points": target_points,
                        }
                        try:
                            row.update(
                                handcrafted_features(
                                    trial,
                                    masks,
                                    baseline_mean,
                                    baseline_std,
                                    audit,
                                )
                            )
                            row.update(
                                visibility_features(
                                    trial,
                                    masks,
                                    args.vg_sampling_rate,
                                    target_points,
                                    audit,
                                )
                            )
                        except (ValueError, TypeError, IndexError, nx.NetworkXException) as exc:
                            reason = (
                                f"window {window_id} failed: {type(exc).__name__}: {exc}"
                            )
                            if args.strict:
                                raise ExtractionError(
                                    f"{subject_id}/{emotion}/{setting.slug}: {reason}"
                                ) from exc
                            audit.warn(f"window_failed_{type(exc).__name__}")
                            continue
                        rows_by_setting[setting.slug].append(row)

            created = sum(len(rows) for rows in rows_by_setting.values()) - participant_rows_before
            elapsed = time.perf_counter() - participant_started
            percent = participant_number * 100.0 / total_participants
            print(
                f"[{participant_number:02d}/{total_participants:02d} | {percent:5.1f}%] "
                f"participant {subject_id}: {created} rows in {elapsed:.1f}s",
                flush=True,
            )

        source_schema_version = str(h5.attrs["schema_version"])
        source_generator_version = str(h5.attrs.get("generator_version", "unknown"))

    script_hash = sha256_file(Path(__file__).resolve())
    elapsed_total = time.perf_counter() - started
    versions = dependency_versions()
    for setting in settings:
        rows = rows_by_setting[setting.slug]
        if not rows:
            raise ExtractionError(
                f"No output rows were produced for {setting.slug}; check filters/window size"
            )
        table = pd.DataFrame(rows)
        feature_columns = sorted(
            column
            for column in table.columns
            if column not in METADATA_COLUMNS and column not in QC_COLUMNS
        )
        family_counts = {
            "handcrafted": sum(column.startswith("hc_") for column in feature_columns),
            "nvg": sum(column.startswith("nvg_") for column in feature_columns),
            "hvg": sum(column.startswith("hvg_") for column in feature_columns),
        }
        if family_counts != EXPECTED_FEATURE_COUNTS:
            raise ExtractionError(
                "Internal feature-schema mismatch: "
                f"expected {EXPECTED_FEATURE_COUNTS}, got {family_counts}"
            )
        zero_std_features = []
        for column in feature_columns:
            values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values) and float(np.std(finite_values, ddof=0)) == 0.0:
                zero_std_features.append(column)
        ordered_columns = METADATA_COLUMNS + QC_COLUMNS + feature_columns
        table = table[ordered_columns].sort_values(
            ["subject_id", "movie_order_index", "window_id"], kind="stable"
        )
        csv_path, manifest_path = paths[setting.slug]
        atomic_write_csv(table, csv_path)
        manifest = {
            "generator": {"name": GENERATOR_NAME, "version": GENERATOR_VERSION},
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command_argv": [str(Path(__file__).resolve()), *sys.argv[1:]],
            "input": {
                "path_hint": input_path.name,
                "sha256": input_hash,
                "schema_name": EXPECTED_SCHEMA,
                "schema_version": source_schema_version,
                "builder_version": source_generator_version,
            },
            "script_sha256": script_hash,
            "output": {
                "csv": csv_path.name,
                "row_count": len(table),
                "column_count": len(table.columns),
                "feature_count": len(feature_columns),
                "feature_counts_by_family": family_counts,
                "feature_columns": feature_columns,
                "zero_standard_deviation_features": zero_std_features,
                "metadata_columns": METADATA_COLUMNS,
                "qc_columns": QC_COLUMNS,
            },
            "parameters": {
                "window_sec": setting.window_sec,
                "overlap_sec": setting.overlap_sec,
                "step_sec": setting.step_sec,
                "vg_sampling_rate_hz": args.vg_sampling_rate,
                "vg_points_per_window": int(
                    round(setting.window_sec * args.vg_sampling_rate)
                ),
                "include_emotions": sorted(include) if include else None,
                "exclude_emotions": sorted(exclude),
                "include_participants": (
                    sorted(included_participants) if included_participants else None
                ),
                "strict": bool(args.strict),
            },
            "methods": {
                "alignment": "intersection of BVP, EDA, and TEMP epoch-time intervals",
                "window_interval": "left-closed and right-open [start, end)",
                "bvp_preprocessing": "NeuroKit2 ppg_process once per full stimulus; no double cleaning",
                "eda_preprocessing": "NeuroKit2 eda_process once per full stimulus; physical units retained",
                "feature_schema": "legacy article-aligned: 29 handcrafted + 48 NVG + 48 HVG descriptors",
                "excluded_legacy_handcrafted": "SKT_mean and EDA_mean (near-zero by construction in the legacy pipeline)",
                "excluded_new_extras": "node count, connected fraction, signal quality/rate, duplicate count/rate, and non-legacy summary descriptors",
                "temperature_preprocessing": "full-stimulus linear detrend for SKT descriptors and VG; raw Celsius retained only for baseline normalisation",
                "temperature_baseline": "z-normalised raw window mean relative to raw BASELINE/STIMULUS participant mean and sample standard deviation",
                "vg_resampling": "scipy resample_poly anti-aliasing followed by exact-length Fourier resampling only when needed",
                "vg_scaling": "within-window z-score; positive affine scaling preserves visibility",
                "nvg_implementation": "ts2vg.NaturalVG",
                "hvg_implementation": "ts2vg.HorizontalVG",
                "degree_entropy": "corrected Shannon entropy of empirical degree frequencies P(k), replacing the legacy degree-value normalisation error",
                "centrality": "exact NetworkX betweenness and closeness",
                "label_policy": "raw SAM 1--9; no thresholding or exclusion",
                "leakage_policy": "structurally constant extra descriptors are excluded by schema; any setting/fold-specific zero-variance filtering must be fitted on training data only",
            },
            "audit": {
                "warning_counts_all_settings": dict(
                    sorted(audit.warning_counts.items())
                ),
                "skipped_trials": audit.skipped_trials,
                "trials_too_short_for_this_setting": audit.no_window_trials.get(
                    setting.slug, []
                ),
                "graph_count_all_settings": dict(audit.graph_count),
                "graph_build_seconds_all_settings": {
                    name: round(value, 6)
                    for name, value in audit.graph_seconds.items()
                },
                "total_run_seconds_all_settings": round(elapsed_total, 6),
            },
            "dependencies": versions,
        }
        atomic_write_json(manifest, manifest_path)
        print(f"Created: {csv_path} ({len(table)} rows, {len(feature_columns)} features)")
        print(f"Manifest: {manifest_path}")
        if zero_std_features:
            print(
                "Zero-SD features in this complete output (exclude inside each "
                "training fold): " + ", ".join(zero_std_features)
            )

    print(f"Completed successfully in {elapsed_total:.1f} seconds.")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except ExtractionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled by user; final output files were not published.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
