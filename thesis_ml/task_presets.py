from __future__ import annotations
from dataclasses import dataclass
from .errors import ExperimentError


@dataclass(frozen=True)
class TaskPreset:
    class_groups: dict[str, tuple[float, ...]]
    excluded_values: tuple[float, ...] = ()
    positive_class: str | None = None


# Presets for targets measured on a 1--9 Self-Assessment Manikin (SAM) scale.
# Dictionary order is the encoded class order (0, 1, ...).
SAM_1_9_PRESETS: dict[str, TaskPreset] = {
    "binary": TaskPreset(
        class_groups={"low": (1, 2, 3, 4), "high": (6, 7, 8, 9)},
        excluded_values=(5,),
        positive_class="high",
    ),
    "ternary": TaskPreset(
        class_groups={
            "low": (1, 2, 3),
            "middle": (4, 5, 6),
            "high": (7, 8, 9),
        },
    ),
    "multiclass": TaskPreset(
        class_groups={f"sam_{value}": (value,) for value in range(1, 10)},
    ),
    "regression": TaskPreset(class_groups={}),
}


TASK_PRESET_SETS: dict[str, dict[str, TaskPreset]] = {
    "sam_1_9": SAM_1_9_PRESETS,
}


def get_task_preset(preset_set: str, kind: str) -> TaskPreset:
    """Return the declared preset or fail before any experiment is run."""

    try:
        presets = TASK_PRESET_SETS[preset_set]
    except KeyError as exc:
        available = ", ".join(sorted(TASK_PRESET_SETS))
        raise ExperimentError(
            f"Unknown task preset set {preset_set!r}; available: {available}"
        ) from exc
    try:
        return presets[kind]
    except KeyError as exc:
        available = ", ".join(sorted(presets))
        raise ExperimentError(
            f"Preset set {preset_set!r} does not define task kind {kind!r}; "
            f"available: {available}"
        ) from exc
