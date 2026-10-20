from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ..domain import CandidateSpec
from ..errors import ExperimentError
from ..utils import json_default


def _supports_dynamic_progress() -> bool:
    if bool(getattr(sys.stderr, "isatty", lambda: False)()):
        return True
    try:
        from IPython import get_ipython

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except (ImportError, NameError):
        return False


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """One updating tqdm bar plus a durable, line-oriented run log."""

    def __init__(
        self,
        total: int,
        target_name: str,
        target_index: int,
        target_count: int,
        run_dir: Path | None,
        enabled: bool = True,
    ) -> None:
        self.total = max(int(total), 1)
        self.target_name = target_name
        self.target_index = target_index
        self.target_count = max(target_count, 1)
        self.enabled = enabled
        self.dynamic = enabled and _supports_dynamic_progress()
        self.run_dir = run_dir
        self.completed = 0
        self.started = time.perf_counter()
        self._bar = tqdm(
            total=self.total,
            desc=f"Target {target_index}/{target_count}: {target_name}",
            unit="fit",
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            disable=not self.dynamic,
        )

    def _write_log(self, level: str, message: str) -> None:
        if self.run_dir is None:
            return
        stamp = datetime.now(timezone.utc).isoformat()
        with (self.run_dir / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{stamp} | {level} | {message}\n")

    def log(self, message: str, *, level: str = "INFO", console: bool = False) -> None:
        self._write_log(level, message)
        if console and self.enabled:
            line = f"{level} | {message}"
            if self.dynamic:
                self._bar.write(line)
            else:
                print(line, flush=True)

    def advance(
        self,
        units: int = 1,
        *,
        phase: str,
        outer_fold: int,
        outer_total: int,
        candidate: str,
        force: bool = False,
    ) -> None:
        increment = min(max(int(units), 0), self.total - self.completed)
        self.completed += increment
        overall = (
            (self.target_index - 1) + self.completed / self.total
        ) / self.target_count
        self._bar.set_postfix_str(
            f"run={overall * 100:.1f}% | LOSO={outer_fold}/{outer_total} | "
            f"{phase} | {candidate}",
            refresh=False,
        )
        if increment:
            self._bar.update(increment)
        elif force and self.dynamic:
            self._bar.refresh()

    def snapshot(self) -> dict[str, Any]:
        now = time.perf_counter()
        fraction = self.completed / self.total
        elapsed = now - self.started
        eta = elapsed * (self.total - self.completed) / self.completed if self.completed else None
        return {
            "target": self.target_name,
            "target_index": self.target_index,
            "target_count": self.target_count,
            "completed_fit_units": self.completed,
            "total_fit_units": self.total,
            "target_percent": 100.0 * fraction,
            "overall_percent": 100.0
            * ((self.target_index - 1) + fraction)
            / self.target_count,
            "session_elapsed_seconds": elapsed,
            "estimated_remaining_seconds": eta,
            "estimated_remaining": _duration(eta),
        }

    def close(self) -> None:
        self._bar.close()


class PhaseProgressReporter:
    """Progress and durable logging for a finite post-processing phase."""

    def __init__(
        self,
        total: int,
        phase_name: str,
        run_dir: Path | None,
        *,
        enabled: bool = True,
        unit: str = "item",
    ) -> None:
        if total <= 0:
            raise ValueError("Phase progress total must be positive")
        self.total = int(total)
        self.phase_name = phase_name
        self.run_dir = run_dir
        self.enabled = enabled
        self.dynamic = enabled and _supports_dynamic_progress()
        self.completed = 0
        self.started = time.perf_counter()
        self._finished = False
        self._bar = tqdm(
            total=self.total,
            desc=phase_name,
            unit=unit,
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            disable=not self.dynamic,
        )

    def _write_log(self, level: str, message: str) -> None:
        if self.run_dir is None:
            return
        stamp = datetime.now(timezone.utc).isoformat()
        with (self.run_dir / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(f"{stamp} | {level} | {message}\n")

    def _console(self, level: str, message: str) -> None:
        if not self.enabled:
            return
        line = f"{level} | {message}"
        if self.dynamic:
            self._bar.write(line)
        else:
            print(line, flush=True)

    def start(self) -> None:
        message = f"phase={self.phase_name} | started | total={self.total}"
        self._write_log("INFO", message)
        self._console("INFO", message)

    def advance(self, units: int = 1, *, detail: str = "") -> None:
        increment = min(max(int(units), 0), self.total - self.completed)
        self.completed += increment
        if detail:
            self._bar.set_postfix_str(detail, refresh=False)
        if increment:
            self._bar.update(increment)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        elapsed = time.perf_counter() - self.started
        message = (
            f"phase={self.phase_name} | complete | "
            f"completed={self.completed}/{self.total} | elapsed={_duration(elapsed)}"
        )
        self._write_log("INFO", message)
        self._console("INFO", message)
        self._bar.close()

    def fail(self, exc: BaseException) -> None:
        if self._finished:
            return
        self._finished = True
        elapsed = time.perf_counter() - self.started
        message = (
            f"phase={self.phase_name} | failed | "
            f"completed={self.completed}/{self.total} | elapsed={_duration(elapsed)} | "
            f"{type(exc).__name__}: {exc}"
        )
        self._write_log("ERROR", message)
        self._console("ERROR", message)
        self._bar.close()


@dataclass
class TargetCheckpointStore:
    run_dir: Path
    target_name: str
    task_kind: str
    enabled: bool

    @property
    def path(self) -> Path:
        return self.run_dir / "checkpoint.sqlite"

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    target TEXT NOT NULL,
                    task_kind TEXT NOT NULL,
                    outer_fold INTEGER NOT NULL,
                    outer_subject_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    candidate_order INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_utc TEXT NOT NULL,
                    PRIMARY KEY (
                        target, task_kind, outer_fold, outer_subject_id,
                        candidate_id, stage
                    )
                )
                """
            )
            connection.commit()
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoint_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_utc TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _load(
        self,
        stage: str,
        outer_fold: int,
        outer_subject: str,
        candidate: CandidateSpec,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT candidate_order, payload_json
                    FROM checkpoints
                    WHERE target=? AND task_kind=? AND outer_fold=?
                      AND outer_subject_id=? AND candidate_id=? AND stage=?
                    """,
                    (
                        self.target_name,
                        self.task_kind,
                        outer_fold,
                        str(outer_subject),
                        candidate.candidate_id,
                        stage,
                    ),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ExperimentError(f"Cannot read checkpoint database: {self.path}") from exc
        if row is None:
            return None
        candidate_order, serialized = row
        if int(candidate_order) != candidate.order:
            raise ExperimentError(
                "Checkpoint candidate order does not match the current configuration "
                f"for {candidate.candidate_id}"
            )
        try:
            return json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ExperimentError(
                f"Invalid checkpoint payload for {candidate.candidate_id}"
            ) from exc

    def _save(
        self,
        stage: str,
        outer_fold: int,
        outer_subject: str,
        candidate: CandidateSpec,
        payload: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        serialized = json.dumps(payload, sort_keys=True, default=json_default)
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO checkpoints (
                        target, task_kind, outer_fold, outer_subject_id,
                        candidate_id, candidate_order, stage, payload_json,
                        updated_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        target, task_kind, outer_fold, outer_subject_id,
                        candidate_id, stage
                    ) DO UPDATE SET
                        candidate_order=excluded.candidate_order,
                        payload_json=excluded.payload_json,
                        updated_utc=excluded.updated_utc
                    """,
                    (
                        self.target_name,
                        self.task_kind,
                        outer_fold,
                        str(outer_subject),
                        candidate.candidate_id,
                        candidate.order,
                        stage,
                        serialized,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise ExperimentError(f"Cannot write checkpoint database: {self.path}") from exc

    def load_inner(
        self,
        outer_fold: int,
        outer_subject: str,
        candidate: CandidateSpec,
    ) -> dict[str, Any] | None:
        return self._load("inner", outer_fold, outer_subject, candidate)

    def save_inner(
        self,
        outer_fold: int,
        outer_subject: str,
        candidate: CandidateSpec,
        payload: dict[str, Any],
    ) -> None:
        self._save("inner", outer_fold, outer_subject, candidate, payload)

    def load_outer(
        self,
        outer_fold: int,
        outer_subject: str,
        candidate: CandidateSpec,
    ) -> dict[str, Any] | None:
        return self._load("outer", outer_fold, outer_subject, candidate)

    def save_outer(
        self,
        outer_fold: int,
        outer_subject: str,
        candidate: CandidateSpec,
        payload: dict[str, Any],
    ) -> None:
        self._save("outer", outer_fold, outer_subject, candidate, payload)

    def write_progress(self, reporter: ProgressReporter, **extra: Any) -> None:
        if not self.enabled:
            return
        payload = json.dumps(
            {"status": "running", **reporter.snapshot(), **extra},
            sort_keys=True,
            default=json_default,
        )
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO checkpoint_metadata (key, value_json, updated_utc)
                    VALUES ('progress', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json=excluded.value_json,
                        updated_utc=excluded.updated_utc
                    """,
                    (payload, datetime.now(timezone.utc).isoformat()),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise ExperimentError(
                f"Cannot update checkpoint progress: {self.path}"
            ) from exc
