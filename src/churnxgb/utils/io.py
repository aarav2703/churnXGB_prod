from __future__ import annotations

from pathlib import Path
import json
import os
import uuid

import joblib
import pandas as pd


def _tmp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def atomic_write_json(path: Path, payload: object) -> Path:
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def atomic_write_csv(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    tmp = _tmp_path(path)
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)
    return path


def atomic_write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> Path:
    tmp = _tmp_path(path)
    df.to_parquet(tmp, **kwargs)
    os.replace(tmp, path)
    return path


def atomic_joblib_dump(model, path: Path) -> Path:
    tmp = _tmp_path(path)
    joblib.dump(model, tmp)
    os.replace(tmp, path)
    return path
