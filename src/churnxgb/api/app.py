from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from churnxgb.inference.contracts import (
    IDENTIFIER_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
    TRAINING_ONLY_COLUMNS,
    build_prediction_output,
)
from churnxgb.pipeline.score import load_model, score_dataframe


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]]


def _repo_root_from_app_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_api_config(repo_root: Path) -> dict:
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {"eval": {"budgets": []}, "mlflow": {"tracking_uri": "sqlite:///mlflow.db"}}

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_request_frame(rows: list[dict[str, Any]], contract: dict) -> pd.DataFrame:
    if len(rows) == 0:
        raise HTTPException(status_code=422, detail="rows must contain at least one item")

    df = pd.DataFrame(rows)
    feature_cols = list(contract["inference_input_columns"])
    id_cols = list(contract["inference_id_columns"])
    allowed_cols = set(feature_cols) | set(id_cols)

    training_only = sorted(set(df.columns) & set(TRAINING_ONLY_COLUMNS))
    if training_only:
        raise HTTPException(
            status_code=422,
            detail=f"Training-only columns are not allowed: {training_only}",
        )

    extra_cols = sorted(set(df.columns) - allowed_cols)
    if extra_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected request columns: {extra_cols}",
        )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required inference columns: {missing}",
        )

    for col in id_cols:
        if col not in df.columns:
            df[col] = None

    return df[id_cols + feature_cols]


def _serialize_prediction_output(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = build_prediction_output(df).copy()

    if "invoice_month" in out.columns:
        out["invoice_month"] = out["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    if "T" in out.columns:
        out["T"] = out["T"].map(
            lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
        )

    out = out.where(pd.notna(out), None)
    return out[PREDICTION_OUTPUT_COLUMNS].to_dict(orient="records")


def create_app(repo_root: Path | None = None) -> FastAPI:
    repo_root = repo_root or _repo_root_from_app_file()
    cfg = _load_api_config(repo_root)
    budgets = [float(x) for x in cfg.get("eval", {}).get("budgets", [])]
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "sqlite:///mlflow.db")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.repo_root = repo_root
        app.state.budgets = budgets
        app.state.model_info = load_model(repo_root, tracking_uri)
        yield

    app = FastAPI(title="ChurnXGB API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        model_info = app.state.model_info
        return {
            "status": "ok",
            "model_name": model_info["model_name"],
            "model_source": model_info["model_source"],
        }

    @app.post("/predict")
    def predict(request: PredictRequest) -> list[dict[str, Any]]:
        model_info = app.state.model_info
        request_df = _prepare_request_frame(request.rows, model_info["contract"])
        scored = score_dataframe(
            request_df,
            model=model_info["model"],
            feature_cols=model_info["feature_cols"],
            contract=model_info["contract"],
            budgets=app.state.budgets,
            model_source=model_info["model_source"],
        )
        return _serialize_prediction_output(scored)

    return app


app = create_app()
