#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "h5py==3.14.0",
#   "numpy==2.2.6",
# ]
# ///
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import h5py
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required package. Use Python 3.11 and install exactly:\n"
        "  python -m pip install numpy==2.2.6 h5py==3.14.0\n"
        f"Original import error: {exc}"
    ) from exc


SCHEMA_NAME = "emognition_empatica_sam"
SCHEMA_VERSION = "1.0.0"
GENERATOR_NAME = "build_emognition_empatica_hdf5.py"
GENERATOR_VERSION = "1.1.0"
DEVICE_NAME = "EMPATICA"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "emognition_study_data"
DEFAULT_OUTPUT = SCRIPT_DIR / "emognition_empatica_sam.h5"

PHASES = ("STIMULUS", "QUESTIONNAIRES", "WASHOUT")
BASELINE_PHASES = ("STIMULUS", "QUESTIONNAIRES")

TIMESTAMP_FORMATS = (
    "%Y-%m-%dT%H:%M:%S:%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)
NAIVE_EPOCH = datetime(1970, 1, 1)

SIGNAL_METADATA: dict[str, dict[str, Any]] = {
    "BVP": {
        "channels": ("bvp",),
        "units": ("device_units",),
        "regular_sampling": True,
        "nominal_rate_hz": 64.0,
    },
    "TEMP": {
        "channels": ("temperature",),
        "units": ("degrees_Celsius",),
        "regular_sampling": True,
        "nominal_rate_hz": 4.0,
    },
    "IBI": {
        "channels": ("inter_beat_interval",),
        "units": ("seconds",),
        "regular_sampling": False,
        "nominal_rate_hz": np.nan,
    },
    "ACC": {
        "channels": ("x", "y", "z"),
        "units": ("raw_1_over_64_g",) * 3,
        "regular_sampling": True,
        "nominal_rate_hz": 32.0,
    },
    "EDA": {
        "channels": ("eda",),
        "units": ("microsiemens",),
        "regular_sampling": True,
        "nominal_rate_hz": 4.0,
    },
}

EMPATICA_FILENAME = re.compile(
    r"^(?P<participant>[^_]+)_(?P<emotion>.+)_"
    r"(?P<phase>STIMULUS|QUESTIONNAIRES|WASHOUT)_EMPATICA\.json$"
)

HDF5_README = """Emognition Empatica + SAM HDF5 schema (v1.0.0)

participants/<id>/metadata
    Participant metadata copied from <id>_QUESTIONNAIRES.json.

participants/<id>/trials/<EMOTION>
    Attributes sam_valence, sam_arousal, and sam_motivation retain the
    original 1--9 ratings. No label thresholding has been applied.

participants/<id>/trials/<EMOTION>/sam
participants/<id>/trials/<EMOTION>/emotion_ratings
    Structured questionnaire values copied without transformation.

participants/<id>/trials/<EMOTION>/phases/<PHASE>
    Attribute available is true only when the original Empatica JSON exists.

.../signals/<SIGNAL>/timestamps_epoch_s
    Seconds from the timezone-naive epoch 1970-01-01 00:00:00. The source
    timestamps have no timezone, so no host-local timezone conversion occurs.

.../signals/<SIGNAL>/timestamps_relative_s
    Seconds relative to the first sample of that signal and phase.

.../signals/<SIGNAL>/values
    Unprocessed float32 sensor values; shape is (samples, channels).

index/trials
    One row per participant/movie with SAM values and phase availability.

index/signals
    One row per available participant/movie/phase/signal dataset.

provenance/source_files
    Relative path, SHA-256, byte size, and kind for every consumed source file.

provenance/missing_expected_files
    Expected Empatica files absent from the original dataset. They are never
    imputed or synthesized.

Minimal Python access example:

    import h5py
    with h5py.File("emognition_empatica_sam.h5", "r") as h5:
        trial = h5["participants/22/trials/AMUSEMENT"]
        valence = int(trial.attrs["sam_valence"])
        signal = trial["phases/STIMULUS/signals/BVP"]
        time_s = signal["timestamps_relative_s"][:]
        bvp = signal["values"][:, 0]
"""


class DatasetBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceRecord:
    relative_path: str
    sha256: str
    size_bytes: int
    kind: str


@dataclass
class BuildStats:
    participant_count: int = 0
    trial_count: int = 0
    empatica_file_count: int = 0
    questionnaire_file_count: int = 0
    signal_dataset_count: int = 0
    total_sensor_samples: int = 0
    source_records: list[SourceRecord] = field(default_factory=list)
    missing_expected_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trial_rows: list[tuple[Any, ...]] = field(default_factory=list)
    signal_rows: list[tuple[Any, ...]] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a validated HDF5 file from original Emognition Empatica "
            "JSON signals and participant SAM questionnaires."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Raw Emognition participant directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output HDF5 path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 numeric dataset compression (default: gzip)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level 0--9 (default: 4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after the new file validates.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected Empatica phase file is absent.",
    )
    parser.add_argument(
        "--validate-only",
        type=Path,
        metavar="HDF5",
        help="Validate an existing generated HDF5 file without rebuilding it.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{GENERATOR_NAME} {GENERATOR_VERSION}",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.compression_level <= 9:
        parser.error("--compression-level must be between 0 and 9")
    return args


def print_build_start(
    input_dir: Path,
    output: Path,
    compression: str,
    compression_level: int,
    strict: bool,
) -> None:
    print("=" * 72)
    print("Emognition Empatica + SAM HDF5 Builder")
    print(f"Generator version: {GENERATOR_VERSION}")
    print(f"Input directory:   {input_dir.resolve()}")
    print(f"Output file:       {output.resolve()}")
    print(
        "Compression:       "
        + (f"gzip (level {compression_level})" if compression == "gzip" else compression)
    )
    print(f"Strict mode:       {'enabled' if strict else 'disabled'}")
    print("-" * 72)
    print("Starting reconstruction from the original JSON files.")
    print("This process reads, converts, hashes, and validates every source file.")
    print("It may take several minutes. Please wait and do not close this window.")
    print("Participant-level progress will be displayed below.")
    print("=" * 72, flush=True)


def print_validation_start(path: Path) -> None:
    print("=" * 72)
    print("Emognition HDF5 validation")
    print(f"File: {path.resolve()}")
    print("Validation may take a short while. Please wait.")
    print("=" * 72, flush=True)


def participant_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def fixed_bytes(value: str, length: int) -> bytes:
    encoded = value.encode("utf-8")
    if len(encoded) > length:
        raise DatasetBuildError(
            f"Value is too long for the HDF5 index field ({length} bytes): {value!r}"
        )
    return encoded


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_json_and_hash(
    path: Path,
    input_dir: Path,
    kind: str,
) -> tuple[dict[str, Any], str, SourceRecord]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetBuildError(f"Input is not UTF-8: {path}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetBuildError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DatasetBuildError(f"Expected a JSON object at the root: {path}")
    relative_path = path.relative_to(input_dir).as_posix()
    return (
        parsed,
        text,
        SourceRecord(relative_path, digest, len(raw), kind),
    )


def parse_source_timestamp(value: Any, context: str) -> tuple[float, str]:
    if not isinstance(value, str):
        raise DatasetBuildError(f"Timestamp is not a string in {context}: {value!r}")
    for timestamp_format in TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(value, timestamp_format)
            seconds = (parsed - NAIVE_EPOCH).total_seconds()
            return seconds, timestamp_format
        except ValueError:
            continue
    raise DatasetBuildError(f"Unsupported timestamp {value!r} in {context}")


def parse_signal_rows(
    rows: Any,
    signal_name: str,
    source_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str], str, int, int]:
    if not isinstance(rows, list):
        raise DatasetBuildError(
            f"Signal {signal_name} is not a list in {source_path}"
        )

    expected_channels = len(SIGNAL_METADATA.get(signal_name, {}).get("channels", ()))
    if not rows:
        channel_count = expected_channels
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, channel_count), dtype=np.float32),
            [],
            "",
            0,
            0,
        )

    timestamps = np.empty(len(rows), dtype=np.float64)
    values_list: list[list[float]] = []
    raw_timestamps: list[str] = []
    timestamp_format = ""
    row_width: int | None = None

    for row_index, row in enumerate(rows):
        context = f"{source_path.name}/{signal_name}/row {row_index}"
        if not isinstance(row, list) or len(row) < 2:
            raise DatasetBuildError(f"Malformed signal row in {context}: {row!r}")
        if row_width is None:
            row_width = len(row)
        elif len(row) != row_width:
            raise DatasetBuildError(
                f"Inconsistent row width in {context}: expected {row_width}, "
                f"received {len(row)}"
            )

        timestamp_value, used_format = parse_source_timestamp(row[0], context)
        if not timestamp_format:
            timestamp_format = used_format
        elif timestamp_format != used_format:
            raise DatasetBuildError(
                f"Mixed timestamp formats within {source_path.name}/{signal_name}"
            )
        timestamps[row_index] = timestamp_value
        raw_timestamps.append(row[0])

        try:
            values_list.append([float(item) for item in row[1:]])
        except (TypeError, ValueError) as exc:
            raise DatasetBuildError(f"Non-numeric signal value in {context}") from exc

    values = np.asarray(values_list, dtype=np.float32)
    if values.ndim != 2:
        raise DatasetBuildError(
            f"Signal values must be two-dimensional: {source_path}/{signal_name}"
        )
    if expected_channels and values.shape[1] != expected_channels:
        raise DatasetBuildError(
            f"Unexpected channel count for {signal_name} in {source_path}: "
            f"expected {expected_channels}, received {values.shape[1]}"
        )

    time_differences = np.diff(timestamps)
    if np.any(time_differences < 0):
        first_bad = int(np.flatnonzero(time_differences < 0)[0])
        raise DatasetBuildError(
            f"Non-monotonic timestamps in {source_path}/{signal_name} "
            f"between rows {first_bad} and {first_bad + 1}"
        )

    duplicate_timestamp_count = int(np.count_nonzero(time_differences == 0))
    nonfinite_value_count = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
    return (
        timestamps,
        values,
        raw_timestamps,
        timestamp_format,
        duplicate_timestamp_count,
        nonfinite_value_count,
    )


def numeric_dataset_options(
    array: np.ndarray,
    compression: str,
    compression_level: int,
) -> dict[str, Any]:
    if array.size == 0 or compression == "none":
        return {"track_times": False}
    options: dict[str, Any] = {
        "compression": compression,
        "shuffle": True,
        "fletcher32": True,
        "track_times": False,
    }
    if compression == "gzip":
        options["compression_opts"] = compression_level
    return options


def write_utf8_dataset(group: h5py.Group, name: str, value: str) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=string_dtype, track_times=False)


def write_json_value(group: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, dict):
        child = group.create_group(name, track_order=True)
        for key in sorted(value):
            write_json_value(child, str(key), value[key])
        return

    if isinstance(value, list):
        if not value:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(
                name,
                shape=(0,),
                dtype=string_dtype,
                track_times=False,
            )
        elif all(isinstance(item, str) for item in value):
            string_dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(
                name,
                data=np.asarray(value, dtype=object),
                dtype=string_dtype,
                track_times=False,
            )
        elif all(isinstance(item, bool) for item in value):
            group.create_dataset(
                name,
                data=np.asarray(value, dtype=np.bool_),
                track_times=False,
            )
        elif all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            group.create_dataset(
                name,
                data=np.asarray(value),
                track_times=False,
            )
        else:
            write_utf8_dataset(group, name, canonical_json(value))
            group[name].attrs["representation"] = "canonical_json"
        return

    if value is None:
        write_utf8_dataset(group, name, "")
        group[name].attrs["source_value_was_null"] = True
    elif isinstance(value, bool):
        group.create_dataset(name, data=np.bool_(value), track_times=False)
    elif isinstance(value, int):
        group.create_dataset(name, data=np.int64(value), track_times=False)
    elif isinstance(value, float):
        group.create_dataset(name, data=np.float64(value), track_times=False)
    elif isinstance(value, str):
        write_utf8_dataset(group, name, value)
    else:
        write_utf8_dataset(group, name, canonical_json(value))
        group[name].attrs["representation"] = "canonical_json"


def questionnaire_by_movie(questionnaire: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    entries = questionnaire.get("questionnaires")
    if not isinstance(entries, list):
        raise DatasetBuildError(f"Missing questionnaire list in {path}")
    result: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("movie"), str):
            raise DatasetBuildError(f"Malformed questionnaire entry {index} in {path}")
        movie = entry["movie"]
        if movie in result:
            raise DatasetBuildError(f"Duplicate questionnaire movie {movie!r} in {path}")
        sam = entry.get("sam")
        if not isinstance(sam, dict):
            raise DatasetBuildError(f"Missing SAM object for {movie!r} in {path}")
        for key in ("VALENCE", "AROUSAL", "MOTIVATION"):
            value = sam.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise DatasetBuildError(f"Invalid SAM {key} for {movie!r} in {path}")
            if not 1 <= float(value) <= 9:
                raise DatasetBuildError(
                    f"SAM {key} is outside 1--9 for {movie!r} in {path}: {value}"
                )
        result[movie] = entry
    return result


def expected_phases_for(emotion: str) -> tuple[str, ...]:
    return BASELINE_PHASES if emotion.upper() == "BASELINE" else PHASES


def discover_empatica_files(
    participant_dir: Path,
    participant_id: str,
) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for path in sorted(participant_dir.glob("*_EMPATICA.json"), key=lambda item: item.name):
        match = EMPATICA_FILENAME.fullmatch(path.name)
        if match is None:
            raise DatasetBuildError(f"Unrecognized Empatica filename: {path}")
        if match.group("participant") != participant_id:
            raise DatasetBuildError(
                f"Participant directory/file mismatch: {participant_dir.name} vs {path.name}"
            )
        key = (match.group("emotion"), match.group("phase"))
        if key in result:
            raise DatasetBuildError(
                f"Duplicate Empatica file for participant {participant_id}, {key}: {path}"
            )
        result[key] = path
    return result


def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    if timestamps.size < 2:
        return np.nan
    positive_differences = np.diff(timestamps)
    positive_differences = positive_differences[positive_differences > 0]
    if positive_differences.size == 0:
        return np.nan
    return float(1.0 / np.median(positive_differences))


def preflight_missing_expected_files(participant_dirs: list[Path]) -> list[str]:
    missing: list[str] = []
    input_dir = participant_dirs[0].parent
    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        questionnaire_path = participant_dir / f"{participant_id}_QUESTIONNAIRES.json"
        if not questionnaire_path.is_file():
            missing.append(questionnaire_path.relative_to(input_dir).as_posix())
            continue
        try:
            questionnaire = json.loads(questionnaire_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DatasetBuildError(
                f"Cannot preflight questionnaire {questionnaire_path}: {exc}"
            ) from exc
        metadata = questionnaire.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("movie_order"), list):
            raise DatasetBuildError(
                f"Cannot preflight invalid movie_order in {questionnaire_path}"
            )
        available = discover_empatica_files(participant_dir, participant_id)
        for emotion in metadata["movie_order"]:
            if not isinstance(emotion, str):
                raise DatasetBuildError(
                    f"Cannot preflight non-string movie name in {questionnaire_path}"
                )
            for phase in expected_phases_for(emotion):
                if (emotion, phase) not in available:
                    filename = f"{participant_id}_{emotion}_{phase}_EMPATICA.json"
                    missing.append(
                        (participant_dir.relative_to(input_dir) / filename).as_posix()
                    )
    return sorted(missing)


def write_signal(
    signals_group: h5py.Group,
    signal_name: str,
    rows: Any,
    source_path: Path,
    compression: str,
    compression_level: int,
    stats: BuildStats,
    participant_id: int,
    emotion: str,
    phase: str,
) -> None:
    (
        timestamps,
        values,
        raw_timestamps,
        timestamp_format,
        duplicate_timestamp_count,
        nonfinite_value_count,
    ) = parse_signal_rows(rows, signal_name, source_path)

    relative_timestamps = (
        timestamps - timestamps[0] if timestamps.size else np.empty(0, dtype=np.float64)
    )
    signal_group = signals_group.create_group(signal_name, track_order=True)
    signal_group.create_dataset(
        "timestamps_epoch_s",
        data=timestamps,
        **numeric_dataset_options(timestamps, compression, compression_level),
    )
    signal_group.create_dataset(
        "timestamps_relative_s",
        data=relative_timestamps,
        **numeric_dataset_options(relative_timestamps, compression, compression_level),
    )
    signal_group.create_dataset(
        "values",
        data=values,
        **numeric_dataset_options(values, compression, compression_level),
    )

    metadata = SIGNAL_METADATA.get(signal_name)
    if metadata is None:
        channels = tuple(f"channel_{index}" for index in range(values.shape[1]))
        units = ("unknown",) * values.shape[1]
        regular_sampling = False
        nominal_rate_hz = np.nan
        stats.warnings.append(
            f"Unknown Empatica signal {signal_name!r} in {source_path.name}; "
            "generic channel metadata was used."
        )
    else:
        channels = metadata["channels"]
        units = metadata["units"]
        regular_sampling = metadata["regular_sampling"]
        nominal_rate_hz = metadata["nominal_rate_hz"]

    estimated_rate_hz = estimate_sampling_rate(timestamps)
    signal_group.attrs["channel_names_json"] = canonical_json(channels)
    signal_group.attrs["units_json"] = canonical_json(units)
    signal_group.attrs["n_samples"] = values.shape[0]
    signal_group.attrs["n_channels"] = values.shape[1]
    signal_group.attrs["value_dtype"] = str(values.dtype)
    signal_group.attrs["regular_sampling"] = bool(regular_sampling)
    signal_group.attrs["nominal_sampling_rate_hz"] = nominal_rate_hz
    signal_group.attrs["estimated_sampling_rate_hz"] = estimated_rate_hz
    signal_group.attrs["source_timestamp_format"] = timestamp_format
    signal_group.attrs["duplicate_timestamp_count"] = duplicate_timestamp_count
    signal_group.attrs["nonfinite_value_count"] = nonfinite_value_count
    if raw_timestamps:
        signal_group.attrs["source_first_timestamp"] = raw_timestamps[0]
        signal_group.attrs["source_last_timestamp"] = raw_timestamps[-1]

    if duplicate_timestamp_count:
        stats.warnings.append(
            f"{source_path.name}/{signal_name} contains "
            f"{duplicate_timestamp_count} duplicate timestamps."
        )
    if nonfinite_value_count:
        stats.warnings.append(
            f"{source_path.name}/{signal_name} contains "
            f"{nonfinite_value_count} non-finite values."
        )

    sample_count = int(values.shape[0])
    start_epoch = float(timestamps[0]) if timestamps.size else np.nan
    end_epoch = float(timestamps[-1]) if timestamps.size else np.nan
    duration = float(relative_timestamps[-1]) if relative_timestamps.size else 0.0
    stats.signal_rows.append(
        (
            participant_id,
            fixed_bytes(emotion, 24),
            fixed_bytes(phase, 16),
            fixed_bytes(signal_name, 16),
            sample_count,
            int(values.shape[1]),
            start_epoch,
            end_epoch,
            duration,
            estimated_rate_hz,
        )
    )
    stats.signal_dataset_count += 1
    stats.total_sensor_samples += sample_count


def write_source_manifest(
    provenance_group: h5py.Group,
    source_records: list[SourceRecord],
) -> str:
    records = sorted(source_records, key=lambda record: record.relative_path)
    source_group = provenance_group.create_group("source_files", track_order=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    source_group.create_dataset(
        "relative_path",
        data=np.asarray([record.relative_path for record in records], dtype=object),
        dtype=string_dtype,
        track_times=False,
    )
    source_group.create_dataset(
        "sha256",
        data=np.asarray([record.sha256.encode("ascii") for record in records], dtype="S64"),
        track_times=False,
    )
    source_group.create_dataset(
        "size_bytes",
        data=np.asarray([record.size_bytes for record in records], dtype=np.int64),
        track_times=False,
    )
    source_group.create_dataset(
        "kind",
        data=np.asarray([fixed_bytes(record.kind, 16) for record in records], dtype="S16"),
        track_times=False,
    )
    digest_input = "".join(
        f"{record.sha256}  {record.relative_path}\n" for record in records
    ).encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def write_indexes(index_group: h5py.Group, stats: BuildStats) -> None:
    trial_dtype = np.dtype(
        [
            ("participant_id", "<i4"),
            ("emotion", "S24"),
            ("movie_order_index", "<i2"),
            ("sam_valence", "<i2"),
            ("sam_arousal", "<i2"),
            ("sam_motivation", "<i2"),
            ("has_stimulus", "?"),
            ("has_questionnaires_phase", "?"),
            ("has_washout", "?"),
        ]
    )
    signal_dtype = np.dtype(
        [
            ("participant_id", "<i4"),
            ("emotion", "S24"),
            ("phase", "S16"),
            ("signal", "S16"),
            ("n_samples", "<i8"),
            ("n_channels", "<i2"),
            ("start_epoch_s", "<f8"),
            ("end_epoch_s", "<f8"),
            ("duration_s", "<f8"),
            ("estimated_sampling_rate_hz", "<f8"),
        ]
    )
    index_group.create_dataset(
        "trials",
        data=np.asarray(stats.trial_rows, dtype=trial_dtype),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
        track_times=False,
    )
    index_group.create_dataset(
        "signals",
        data=np.asarray(stats.signal_rows, dtype=signal_dtype),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
        track_times=False,
    )


def build_hdf5(
    input_dir: Path,
    output_path: Path,
    compression: str,
    compression_level: int,
    strict: bool,
) -> BuildStats:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise DatasetBuildError(f"Input directory does not exist: {input_dir}")

    participant_dirs = sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=participant_sort_key,
    )
    if not participant_dirs:
        raise DatasetBuildError(f"No participant directories found in {input_dir}")

    if strict:
        preflight_missing = preflight_missing_expected_files(participant_dirs)
        if preflight_missing:
            preview = "\n".join(preflight_missing[:20])
            raise DatasetBuildError(
                f"Strict mode found {len(preflight_missing)} missing expected "
                f"Empatica files before build. First entries:\n{preview}"
            )

    stats = BuildStats(participant_count=len(participant_dirs))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"\nDiscovered {len(participant_dirs)} participant folders. "
        "Beginning sensor conversion...",
        flush=True,
    )

    with h5py.File(output_path, "w", libver="earliest", track_order=True) as h5:
        h5.attrs["schema_name"] = SCHEMA_NAME
        h5.attrs["schema_version"] = SCHEMA_VERSION
        h5.attrs["generator_name"] = GENERATOR_NAME
        h5.attrs["generator_version"] = GENERATOR_VERSION
        h5.attrs["source_device"] = DEVICE_NAME
        h5.attrs["sam_policy"] = "raw_1_to_9_no_thresholding"
        h5.attrs["sensor_value_policy"] = "unprocessed_float32"
        h5.attrs["source_timestamp_timezone"] = "unspecified"
        h5.attrs["timestamp_basis"] = (
            "seconds since 1970-01-01 00:00:00 using timezone-naive source "
            "timestamps; no local/UTC conversion"
        )
        write_utf8_dataset(h5, "README", HDF5_README)

        participants_group = h5.create_group("participants", track_order=True)

        for participant_number, participant_dir in enumerate(participant_dirs, start=1):
            participant_id_text = participant_dir.name
            empatica_file_total = len(list(participant_dir.glob("*_EMPATICA.json")))
            progress_percent = 100.0 * participant_number / len(participant_dirs)
            print(
                f"[{participant_number:>2}/{len(participant_dirs)} | "
                f"{progress_percent:>6.2f}%] Participant {participant_id_text}: "
                f"processing {empatica_file_total} Empatica files...",
                flush=True,
            )
            try:
                participant_id = int(participant_id_text)
            except ValueError as exc:
                raise DatasetBuildError(
                    f"Participant directory is not an integer: {participant_dir}"
                ) from exc

            questionnaire_path = participant_dir / f"{participant_id_text}_QUESTIONNAIRES.json"
            if not questionnaire_path.is_file():
                raise DatasetBuildError(
                    f"Missing participant questionnaire: {questionnaire_path}"
                )
            questionnaire, questionnaire_text, source_record = read_json_and_hash(
                questionnaire_path, input_dir, "questionnaire"
            )
            stats.source_records.append(source_record)
            stats.questionnaire_file_count += 1
            entries_by_movie = questionnaire_by_movie(questionnaire, questionnaire_path)

            metadata = questionnaire.get("metadata")
            if not isinstance(metadata, dict):
                raise DatasetBuildError(f"Missing metadata object in {questionnaire_path}")
            if int(metadata.get("id", participant_id)) != participant_id:
                raise DatasetBuildError(
                    f"Metadata participant ID mismatch in {questionnaire_path}"
                )

            movie_order = metadata.get("movie_order")
            if not isinstance(movie_order, list) or not all(
                isinstance(movie, str) for movie in movie_order
            ):
                raise DatasetBuildError(f"Invalid movie_order in {questionnaire_path}")
            if set(movie_order) != set(entries_by_movie):
                raise DatasetBuildError(
                    f"movie_order and questionnaire movies differ in {questionnaire_path}"
                )

            empatica_files = discover_empatica_files(participant_dir, participant_id_text)
            extra_emotions = sorted(
                {emotion for emotion, _phase in empatica_files} - set(entries_by_movie)
            )
            if extra_emotions:
                raise DatasetBuildError(
                    f"Empatica files have movies absent from questionnaire for participant "
                    f"{participant_id}: {extra_emotions}"
                )

            participant_group = participants_group.create_group(
                participant_id_text, track_order=True
            )
            participant_group.attrs["participant_id"] = participant_id
            participant_group.attrs["questionnaire_source_relative_path"] = (
                source_record.relative_path
            )
            participant_group.attrs["questionnaire_source_sha256"] = source_record.sha256
            write_utf8_dataset(
                participant_group, "source_questionnaire_json", questionnaire_text
            )
            metadata_group = participant_group.create_group("metadata", track_order=True)
            for key in sorted(metadata):
                write_json_value(metadata_group, str(key), metadata[key])

            trials_group = participant_group.create_group("trials", track_order=True)
            for movie_order_index, emotion in enumerate(movie_order):
                entry = entries_by_movie[emotion]
                sam = entry["sam"]
                emotions = entry.get("emotions", {})
                if not isinstance(emotions, dict):
                    raise DatasetBuildError(
                        f"Invalid emotions object for {participant_id}/{emotion}"
                    )

                trial_group = trials_group.create_group(emotion, track_order=True)
                trial_group.attrs["emotion"] = emotion
                trial_group.attrs["movie_order_index"] = movie_order_index
                trial_group.attrs["sam_valence"] = int(sam["VALENCE"])
                trial_group.attrs["sam_arousal"] = int(sam["AROUSAL"])
                trial_group.attrs["sam_motivation"] = int(sam["MOTIVATION"])

                sam_group = trial_group.create_group("sam", track_order=True)
                for key in sorted(sam):
                    write_json_value(sam_group, key, sam[key])
                emotion_ratings_group = trial_group.create_group(
                    "emotion_ratings", track_order=True
                )
                for key in sorted(emotions):
                    write_json_value(emotion_ratings_group, key, emotions[key])

                phases_group = trial_group.create_group("phases", track_order=True)
                phase_presence: dict[str, bool] = {}
                for phase in expected_phases_for(emotion):
                    source_path = empatica_files.get((emotion, phase))
                    phase_group = phases_group.create_group(phase, track_order=True)
                    expected_filename = (
                        f"{participant_id_text}_{emotion}_{phase}_EMPATICA.json"
                    )
                    expected_relative_path = (
                        participant_dir.relative_to(input_dir) / expected_filename
                    ).as_posix()
                    phase_group.attrs["expected_source_relative_path"] = (
                        expected_relative_path
                    )
                    phase_presence[phase] = source_path is not None

                    if source_path is None:
                        phase_group.attrs["available"] = False
                        stats.missing_expected_files.append(expected_relative_path)
                        continue

                    phase_group.attrs["available"] = True
                    sensor_data, _sensor_text, sensor_record = read_json_and_hash(
                        source_path, input_dir, "empatica"
                    )
                    stats.source_records.append(sensor_record)
                    stats.empatica_file_count += 1
                    phase_group.attrs["source_relative_path"] = sensor_record.relative_path
                    phase_group.attrs["source_sha256"] = sensor_record.sha256
                    signals_group = phase_group.create_group("signals", track_order=True)
                    for signal_name in sorted(sensor_data):
                        write_signal(
                            signals_group=signals_group,
                            signal_name=signal_name,
                            rows=sensor_data[signal_name],
                            source_path=source_path,
                            compression=compression,
                            compression_level=compression_level,
                            stats=stats,
                            participant_id=participant_id,
                            emotion=emotion,
                            phase=phase,
                        )

                stats.trial_rows.append(
                    (
                        participant_id,
                        fixed_bytes(emotion, 24),
                        movie_order_index,
                        int(sam["VALENCE"]),
                        int(sam["AROUSAL"]),
                        int(sam["MOTIVATION"]),
                        phase_presence.get("STIMULUS", False),
                        phase_presence.get("QUESTIONNAIRES", False),
                        phase_presence.get("WASHOUT", False),
                    )
                )
                stats.trial_count += 1

        print(
            "Sensor conversion finished. Creating indexes and provenance manifest...",
            flush=True,
        )

        index_group = h5.create_group("index", track_order=True)
        write_indexes(index_group, stats)

        provenance_group = h5.create_group("provenance", track_order=True)
        source_tree_digest = write_source_manifest(
            provenance_group, stats.source_records
        )
        provenance_group.attrs["source_tree_sha256"] = source_tree_digest
        provenance_group.attrs["source_root_hint"] = "emognition_study_data"
        provenance_group.attrs["generator_source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        provenance_group.attrs["build_parameters_json"] = canonical_json(
            {
                "compression": compression,
                "compression_level": compression_level,
                "strict": strict,
            }
        )
        provenance_group.attrs["build_environment_json"] = canonical_json(
            {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "h5py": h5py.__version__,
                "hdf5": h5py.version.hdf5_version,
            }
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        provenance_group.create_dataset(
            "missing_expected_files",
            data=np.asarray(sorted(stats.missing_expected_files), dtype=object),
            dtype=string_dtype,
            track_times=False,
        )
        provenance_group.create_dataset(
            "warnings",
            data=np.asarray(sorted(stats.warnings), dtype=object),
            dtype=string_dtype,
            track_times=False,
        )

        h5.attrs["participant_count"] = stats.participant_count
        h5.attrs["trial_count"] = stats.trial_count
        h5.attrs["empatica_file_count"] = stats.empatica_file_count
        h5.attrs["questionnaire_file_count"] = stats.questionnaire_file_count
        h5.attrs["signal_dataset_count"] = stats.signal_dataset_count
        h5.attrs["total_sensor_samples"] = stats.total_sensor_samples
        h5.attrs["missing_expected_file_count"] = len(stats.missing_expected_files)
        h5.attrs["warning_count"] = len(stats.warnings)
        h5.flush()

    return stats


def decode_fixed(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    return str(value)


def validate_hdf5(path: Path) -> dict[str, int]:
    path = path.resolve()
    if not path.is_file():
        raise DatasetBuildError(f"HDF5 file does not exist: {path}")

    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_name") != SCHEMA_NAME:
            raise DatasetBuildError(
                f"Unexpected schema_name in {path}: {h5.attrs.get('schema_name')!r}"
            )
        if h5.attrs.get("schema_version") != SCHEMA_VERSION:
            raise DatasetBuildError(
                f"Unsupported schema_version in {path}: "
                f"{h5.attrs.get('schema_version')!r}"
            )
        for required_path in (
            "README",
            "participants",
            "index/trials",
            "index/signals",
            "provenance/source_files/relative_path",
            "provenance/source_files/sha256",
            "provenance/missing_expected_files",
        ):
            if required_path not in h5:
                raise DatasetBuildError(f"Missing required HDF5 object: {required_path}")

        participant_count = len(h5["participants"])
        trials = h5["index/trials"]
        signals = h5["index/signals"]
        if participant_count != int(h5.attrs["participant_count"]):
            raise DatasetBuildError("Participant count does not match root metadata")
        if len(trials) != int(h5.attrs["trial_count"]):
            raise DatasetBuildError("Trial index length does not match root metadata")
        if len(signals) != int(h5.attrs["signal_dataset_count"]):
            raise DatasetBuildError("Signal index length does not match root metadata")

        total_samples = 0
        for row in trials:
            participant = str(int(row["participant_id"]))
            emotion = decode_fixed(row["emotion"])
            trial_path = f"participants/{participant}/trials/{emotion}"
            if trial_path not in h5:
                raise DatasetBuildError(f"Trial index points to missing group: {trial_path}")
            trial_group = h5[trial_path]
            if int(trial_group.attrs["sam_valence"]) != int(row["sam_valence"]):
                raise DatasetBuildError(f"SAM valence mismatch at {trial_path}")
            if int(trial_group.attrs["sam_arousal"]) != int(row["sam_arousal"]):
                raise DatasetBuildError(f"SAM arousal mismatch at {trial_path}")

        for row in signals:
            participant = str(int(row["participant_id"]))
            emotion = decode_fixed(row["emotion"])
            phase = decode_fixed(row["phase"])
            signal = decode_fixed(row["signal"])
            signal_path = (
                f"participants/{participant}/trials/{emotion}/phases/"
                f"{phase}/signals/{signal}"
            )
            if signal_path not in h5:
                raise DatasetBuildError(
                    f"Signal index points to missing group: {signal_path}"
                )
            signal_group = h5[signal_path]
            timestamps = signal_group["timestamps_epoch_s"]
            relative = signal_group["timestamps_relative_s"]
            values = signal_group["values"]
            expected_samples = int(row["n_samples"])
            expected_channels = int(row["n_channels"])
            if timestamps.shape != (expected_samples,):
                raise DatasetBuildError(f"Timestamp shape mismatch at {signal_path}")
            if relative.shape != (expected_samples,):
                raise DatasetBuildError(f"Relative timestamp shape mismatch at {signal_path}")
            if values.shape != (expected_samples, expected_channels):
                raise DatasetBuildError(f"Value shape mismatch at {signal_path}")
            if expected_samples:
                if not np.isclose(float(relative[0]), 0.0, atol=1e-9):
                    raise DatasetBuildError(
                        f"Relative timestamps do not start at zero: {signal_path}"
                    )
                timestamp_values = timestamps[:]
                if np.any(np.diff(timestamp_values) < 0):
                    raise DatasetBuildError(
                        f"Non-monotonic stored timestamps: {signal_path}"
                    )
            total_samples += expected_samples

        if total_samples != int(h5.attrs["total_sensor_samples"]):
            raise DatasetBuildError("Total sample count does not match root metadata")

        source_files = h5["provenance/source_files"]
        source_count = len(source_files["relative_path"])
        if not (
            source_count
            == len(source_files["sha256"])
            == len(source_files["size_bytes"])
            == len(source_files["kind"])
        ):
            raise DatasetBuildError("Source manifest columns have different lengths")
        hashes = source_files["sha256"][:]
        if any(len(decode_fixed(value)) != 64 for value in hashes):
            raise DatasetBuildError("Source manifest contains an invalid SHA-256")

        return {
            "participants": participant_count,
            "trials": len(trials),
            "signals": len(signals),
            "samples": total_samples,
            "source_files": source_count,
            "missing_expected_files": len(h5["provenance/missing_expected_files"]),
            "warnings": len(h5["provenance/warnings"]),
        }


def publish_build(
    input_dir: Path,
    output: Path,
    compression: str,
    compression_level: int,
    strict: bool,
    overwrite: bool,
) -> tuple[BuildStats, dict[str, int]]:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise DatasetBuildError(
            f"Output already exists: {output}\nUse --overwrite to replace it safely."
        )
    temporary = output.with_name(f".{output.name}.building")
    if temporary.exists():
        temporary.unlink()

    try:
        stats = build_hdf5(
            input_dir=input_dir,
            output_path=temporary,
            compression=compression,
            compression_level=compression_level,
            strict=strict,
        )
        print("Writing finished. Validating the complete HDF5 file...", flush=True)
        validation = validate_hdf5(temporary)
        print("Validation passed. Publishing the final output atomically...", flush=True)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return stats, validation


def print_validation(path: Path, summary: dict[str, int]) -> None:
    print(f"Validated: {path.resolve()}")
    print(f"  Participants:           {summary['participants']}")
    print(f"  Questionnaire trials:   {summary['trials']}")
    print(f"  Signal datasets:        {summary['signals']}")
    print(f"  Total sensor samples:   {summary['samples']}")
    print(f"  Consumed source files:  {summary['source_files']}")
    print(f"  Missing expected files: {summary['missing_expected_files']}")
    print(f"  Data warnings:          {summary['warnings']}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate_only is not None:
            print_validation_start(args.validate_only)
            summary = validate_hdf5(args.validate_only)
            print_validation(args.validate_only, summary)
            return 0

        print_build_start(
            input_dir=args.input_dir,
            output=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            strict=args.strict,
        )
        stats, summary = publish_build(
            input_dir=args.input_dir,
            output=args.output,
            compression=args.compression,
            compression_level=args.compression_level,
            strict=args.strict,
            overwrite=args.overwrite,
        )
        print_validation(args.output, summary)
        if stats.missing_expected_files:
            print(
                "\nNotice: missing source files were recorded, not synthesized. "
                "See /provenance/missing_expected_files in the HDF5 file."
            )
            for relative_path in sorted(stats.missing_expected_files)[:10]:
                print(f"  missing: {relative_path}")
            if len(stats.missing_expected_files) > 10:
                print(f"  ... and {len(stats.missing_expected_files) - 10} more")
        print("\nBuild completed successfully.")
        return 0
    except (DatasetBuildError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
