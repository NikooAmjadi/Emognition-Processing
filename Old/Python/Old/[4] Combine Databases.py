
import os
import json
import h5py
import numpy as np

 # آدرس فایل ها را از اینجا تغییر دهید
QUESTIONNAIRE_ROOT = "./emognition"

EMPATICA_DB = "emognition_empatica.h5"
MUSE_DB = "emognition_muse.h5"
SAMSUNG_DB = "emognition_samsung_watch.h5"

OUTPUT_DB = "emognition_complete.h5"


def copy_hdf5_group(source_group, target_group):
    """
    Recursive copy of complete HDF5 hierarchy
    """

    for key in source_group.keys():

        item = source_group[key]

        if isinstance(item, h5py.Group):

            new_group = target_group.create_group(key)

            copy_hdf5_group(item, new_group)

        elif isinstance(item, h5py.Dataset):

            source_group.copy(key, target_group)


def save_string(group, name, value):

    if value is None:
        value = ""

    group.create_dataset(
        name,
        data=np.bytes_(str(value))
    )


print("Creating unified database...")

with h5py.File(OUTPUT_DB, "w") as out_h5:

    print("\nCopying EMPATICA...")

    with h5py.File(EMPATICA_DB, "r") as emp_h5:

        for participant in emp_h5.keys():

            participant_group = out_h5.require_group(
                participant
            )

            devices_group = participant_group.require_group(
                "devices"
            )

            empatica_group = devices_group.create_group(
                "EMPATICA"
            )

            copy_hdf5_group(
                emp_h5[participant],
                empatica_group
            )

    print("Copying MUSE...")

    with h5py.File(MUSE_DB, "r") as muse_h5:

        for participant in muse_h5.keys():

            participant_group = out_h5.require_group(
                participant
            )

            devices_group = participant_group.require_group(
                "devices"
            )

            muse_group = devices_group.create_group(
                "MUSE"
            )

            copy_hdf5_group(
                muse_h5[participant],
                muse_group
            )

    print("Copying SAMSUNG WATCH...")

    with h5py.File(SAMSUNG_DB, "r") as samsung_h5:

        for participant in samsung_h5.keys():

            participant_group = out_h5.require_group(
                participant
            )

            devices_group = participant_group.require_group(
                "devices"
            )

            samsung_group = devices_group.create_group(
                "SAMSUNG_WATCH"
            )

            copy_hdf5_group(
                samsung_h5[participant],
                samsung_group
            )

    print("Adding questionnaires...")

    for participant in os.listdir(
        QUESTIONNAIRE_ROOT
    ):

        participant_path = os.path.join(
            QUESTIONNAIRE_ROOT,
            participant
        )

        if not os.path.isdir(
            participant_path
        ):
            continue

        questionnaire_file = os.path.join(
            participant_path,
            f"{participant}_QUESTIONNAIRES.json"
        )

        if not os.path.exists(
            questionnaire_file
        ):
            print(
                f"Questionnaire missing: "
                f"{participant}"
            )
            continue

        print(
            f"Processing questionnaire: "
            f"{participant}"
        )

        with open(
            questionnaire_file,
            "r",
            encoding="utf-8"
        ) as f:

            questionnaire_data = json.load(f)

        participant_group = out_h5.require_group(
            participant
        )

        metadata = questionnaire_data.get(
            "metadata",
            {}
        )

        metadata_group = participant_group.require_group(
            "metadata"
        )

        for key, value in metadata.items():

            if isinstance(
                value,
                (dict, list)
            ):
                continue

            try:

                if isinstance(value, bool):

                    metadata_group.create_dataset(
                        key,
                        data=int(value)
                    )

                elif isinstance(
                    value,
                    (int, float)
                ):

                    metadata_group.create_dataset(
                        key,
                        data=value
                    )

                else:

                    save_string(
                        metadata_group,
                        key,
                        value
                    )

            except Exception:
                pass

        movies_seen = metadata.get(
            "movies_seen_before_study",
            {}
        )

        movies_group = metadata_group.require_group(
            "movies_seen_before_study"
        )

        for emotion, score in movies_seen.items():

            movies_group.create_dataset(
                emotion,
                data=score
            )

        movie_order = metadata.get(
            "movie_order",
            []
        )

        order_group = metadata_group.require_group(
            "movie_order"
        )

        for idx, movie in enumerate(
            movie_order
        ):

            save_string(
                order_group,
                str(idx),
                movie
            )

        questionnaires_group = (
            participant_group.require_group(
                "questionnaires"
            )
        )

        for questionnaire in questionnaire_data.get(
            "questionnaires",
            []
        ):

            movie_name = questionnaire[
                "movie"
            ]

            movie_group = (
                questionnaires_group.require_group(
                    movie_name
                )
            )

            sam_group = movie_group.require_group(
                "sam"
            )

            sam_data = questionnaire.get(
                "sam",
                {}
            )

            for key, value in sam_data.items():

                sam_group.create_dataset(
                    key,
                    data=value
                )

            emotions_group = (
                movie_group.require_group(
                    "emotions"
                )
            )

            emotion_data = questionnaire.get(
                "emotions",
                {}
            )

            for emotion, value in (
                emotion_data.items()
            ):

                emotions_group.create_dataset(
                    emotion,
                    data=value
                )

print("\nDone!")
print(f"Unified database saved as:")
print(OUTPUT_DB)
