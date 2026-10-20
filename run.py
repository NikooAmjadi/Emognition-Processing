#!/usr/bin/env python3
"""Run the experiment configured in config.py."""

from __future__ import annotations

import sys
from pathlib import Path

from config import CONFIG
from thesis_ml.errors import ExperimentError
from thesis_ml.pipeline import run_pipeline


def main() -> int:
    try:
        run_pipeline(CONFIG, Path(__file__).resolve().parent)
        return 0
    except (ExperimentError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
