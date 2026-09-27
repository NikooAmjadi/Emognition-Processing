
import os
import json
import h5py
import numpy as np
from datetime import datetime

 # آدرس فایل ها را از اینجا تغییر دهید
ROOT_DIR = r"./emognition/"
OUTPUT_HDF5 = "emognition_muse.h5"

TARGET_DEVICE = "MUSE"

EMOTIONS = [
    "BASELINE",
    "ANGER",
    "ENTHUSIASM",
    "LIKING",
    "FEAR",
    "AMUSEMENT",
    "SADNESS",
    "NEUTRAL",
    "AWE",
    "DISGUST",
    "SURPRISE"
]

def parse_timestamp(ts):
    """
    Convert timestamp string to unix timestamp
    """

    try:
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        return dt.timestamp()

    except Exception:
        return np.nan


print("Creating MUSE HDF5 dataset...")

with h5py.File(OUTPUT_HDF5, "w") as hdf:

    for participant in os.listdir(ROOT_DIR):

        participant_path = os.path.join(
            ROOT_DIR,
            participant
        )

        if not os.path.isdir(participant_path):
            continue

        print(f"\nParticipant: {participant}")

        participant_group = hdf.require_group(
            participant
        )

        for filename in os.listdir(participant_path):

            if not filename.endswith(".json"):
                continue

            if "QUESTIONNAIRES" in filename:
                continue

            file_without_ext = filename[:-5]

            try:

                parts = file_without_ext.split("_")

                participant_id = parts[0]

                emotion_name = parts[1]

                phase_name = parts[2]

                device_name = "_".join(parts[3:])

            except Exception:

                print(
                    f"Filename parsing error: "
                    f"{filename}"
                )

                continue

            if device_name != TARGET_DEVICE:
                continue

            print(
                f"  Processing -> "
                f"Emotion: {emotion_name} | "
                f"Phase: {phase_name}"
            )

            file_path = os.path.join(
                participant_path,
                filename
            )

            try:

                with open(file_path, "r") as f:
                    data = json.load(f)

            except Exception as e:

                print(f"JSON read error: {filename}")
                print(e)

                continue

            if "TimeStamp" not in data:

                print(
                    f"Missing TimeStamp in {filename}"
                )

                continue

            timestamps_raw = data["TimeStamp"]

            if len(timestamps_raw) == 0:

                print(
                    f"Empty TimeStamp in {filename}"
                )

                continue

            timestamps_absolute = np.array(
                [
                    parse_timestamp(ts)
                    for ts in timestamps_raw
                ],
                dtype=np.float64
            )

            timestamps_relative = (
                timestamps_absolute
                - timestamps_absolute[0]
            )

            emotion_group = participant_group.require_group(
                emotion_name
            )

            phase_group = emotion_group.require_group(
                phase_name
            )

            for signal_name, signal_values in data.items():

                if signal_name == "TimeStamp":
                    continue

                if len(signal_values) == 0:
                    continue

                try:

                    values = np.array(
                        signal_values,
                        dtype=np.float32
                    )

                    signal_group = phase_group.require_group(
                        signal_name
                    )

                    for dataset_name in [
                        "timestamps_absolute",
                        "timestamps_relative",
                        "values"
                    ]:

                        if dataset_name in signal_group:
                            del signal_group[dataset_name]

                    signal_group.create_dataset(
                        "timestamps_absolute",
                        data=timestamps_absolute,
                        compression="gzip"
                    )

                    signal_group.create_dataset(
                        "timestamps_relative",
                        data=timestamps_relative,
                        compression="gzip"
                    )

                    signal_group.create_dataset(
                        "values",
                        data=values,
                        compression="gzip"
                    )

                    print(
                        f"    Saved: {signal_name} "
                        f"shape={values.shape}"
                    )

                except Exception as e:

                    print(
                        f"Signal processing error: "
                        f"{signal_name}"
                    )

                    print(e)

print("\nMUSE HDF5 dataset creation completed.")
