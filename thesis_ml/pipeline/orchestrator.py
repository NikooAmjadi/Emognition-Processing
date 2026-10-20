from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config_types import AppConfig
from ..data import prepare_datasets
from ..domain import RunResults
from ..evaluation import expand_candidate_specs, run_nested_target
from ..evaluation.runtime import PhaseProgressReporter
from ..models import available_models
from ..output import OutputManager
from ..reports import (
    paired_statistical_test_count,
    paired_statistical_tests,
    write_descriptive_reports,
)
from ..tasks import create_task
from ..utils import atomic_write_csv, atomic_write_json, set_global_seed


def run_pipeline(config: AppConfig, project_dir: Path) -> Path:
    config.validate()
    requested_models = set(config.models.enabled) | {
        candidate.model_name for candidate in config.selection.candidates
    }
    unknown_models = requested_models - set(available_models())
    if unknown_models:
        from ..errors import ExperimentError

        raise ExperimentError(f"Unknown models: {sorted(unknown_models)}")
    set_global_seed(config.evaluation.seed)
    task = create_task(config.task)
    datasets, participant_rows, input_rows = prepare_datasets(config, project_dir)
    candidate_specs = expand_candidate_specs(datasets, config)
    output = OutputManager(config, datasets)
    started = time.perf_counter()
    started_utc = output.previous_manifest.get(
        "started_utc", datetime.now(timezone.utc).isoformat()
    )
    output.write_manifest(
        "running",
        datasets,
        started_utc=started_utc,
        target_count=len(config.task.targets),
        candidate_count=len(candidate_specs),
    )
    output.write_eligibility(participant_rows, input_rows)
    write_descriptive_reports(output.run_dir, datasets, task, config)
    results = RunResults()
    print("=" * 78)
    print(
        "Dataset-agnostic nested run: "
        f"{len(config.task.targets)} targets; {len(candidate_specs)} independently "
        "evaluated candidates per target"
    )
    print(f"Task: {config.task.kind}; outer=LOSO; inner=grouped CV")
    print(f"Output: {output.run_dir}")
    if output.resumed:
        print("Resume: validated configuration and input hashes; loading checkpoints")
    print("=" * 78, flush=True)
    try:
        for index, target_name in enumerate(config.task.targets, start=1):
            results.add_nested(
                run_nested_target(
                    datasets,
                    candidate_specs,
                    target_name,
                    task,
                    config,
                    run_dir=output.run_dir,
                    target_index=index,
                    target_count=len(config.task.targets),
                )
            )
            output.write_results(
                results,
                {
                    "status": "running",
                    "completed_targets": index,
                    "total_targets": len(config.task.targets),
                    "elapsed_seconds": time.perf_counter() - started,
                    "last_experiment_id": results.summaries[-1]["experiment_id"],
                },
            )
    except (Exception, KeyboardInterrupt) as exc:
        print("", flush=True)
        with (output.run_dir / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(
                f"{datetime.now(timezone.utc).isoformat()} | ERROR | "
                f"{type(exc).__name__}: {exc}\n"
            )
        output.write_manifest(
            "interrupted",
            datasets,
            started_utc=started_utc,
            interrupted_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started,
            target_count=len(config.task.targets),
            candidate_count=len(candidate_specs),
            error_type=type(exc).__name__,
            error_message=str(exc),
            resumable=config.output.save_checkpoints,
        )
        raise
    if config.analyses.statistical_tests:
        prediction_frame = pd.DataFrame(results.predictions)
        statistical_total = paired_statistical_test_count(prediction_frame, task)
        statistical_progress = (
            PhaseProgressReporter(
                total=statistical_total,
                phase_name="Statistical tests",
                run_dir=output.run_dir,
                enabled=config.evaluation.show_progress,
                unit="test",
            )
            if statistical_total > 0
            else None
        )
        if statistical_progress is not None:
            statistical_progress.start()
        try:
            tests = paired_statistical_tests(
                prediction_frame,
                task,
                config.analyses.statistical_permutation_repeats,
                config.evaluation.seed,
                progress_callback=(
                    (
                        lambda _completed, _total, detail: statistical_progress.advance(
                            detail=detail
                        )
                    )
                    if statistical_progress is not None
                    else None
                ),
            )
        except (Exception, KeyboardInterrupt) as exc:
            if statistical_progress is not None:
                statistical_progress.fail(exc)
            raise
        else:
            if statistical_progress is not None:
                statistical_progress.finish()
        if tests.empty:
            atomic_write_json(
                {
                    "status": "not_applicable",
                    "reason": (
                        "No eligible predeclared candidate pair was available within "
                        "a common model, representation, or input comparison family."
                    ),
                },
                output.run_dir / "statistical_tests_status.json",
            )
        else:
            atomic_write_csv(tests, output.run_dir / "paired_statistical_tests.csv")
    elapsed = time.perf_counter() - started
    finished_utc = datetime.now(timezone.utc).isoformat()
    output.write_results(
        results,
        {
            "status": "complete",
            "completed_targets": len(config.task.targets),
            "total_targets": len(config.task.targets),
            "elapsed_seconds": elapsed,
            "finished_utc": finished_utc,
        },
    )
    output.write_manifest(
        "complete",
        datasets,
        started_utc=started_utc,
        finished_utc=finished_utc,
        elapsed_seconds=elapsed,
        target_count=len(config.task.targets),
        candidate_count=len(candidate_specs),
        reportable_final_run=bool(
            not config.evaluation.quick
            and config.evaluation.max_outer_folds == 0
            and config.evaluation.bootstrap_repeats > 0
        ),
    )
    output.finalize_checkpoint()
    print(f"Complete in {elapsed / 60:.1f} minutes: {output.run_dir}", flush=True)
    return output.run_dir
