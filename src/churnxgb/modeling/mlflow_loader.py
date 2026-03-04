from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd


def load_promoted_sklearn_model_from_run_id(
    repo_root: Path, run_id: str, tracking_uri: str
) -> Any:
    """
    Load the sklearn-flavor model for a given run_id.

    This guarantees we can call predict_proba() and get proper probabilities.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def predict_proba_1(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Return p(class=1) from a sklearn-like model supporting predict_proba.
    """
    p = model.predict_proba(X)[:, 1]
    return pd.Series(p, index=X.index, dtype=float)
