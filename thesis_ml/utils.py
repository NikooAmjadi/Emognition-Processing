from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(map(str, (base_seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._+-]+", "_", str(value)).strip("_.")
    return token or "run"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
