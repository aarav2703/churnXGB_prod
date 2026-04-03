from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from churnxgb.api.artifacts import (
    load_feature_importance,
    load_model_comparison,
    load_model_summary,
)
from churnxgb.api.serializers import serialize_records


router = APIRouter(tags=["summary"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    model_info = request.app.state.model_info
    return {
        "status": "ok",
        "model_name": model_info["model_name"],
        "model_source": model_info["model_source"],
        "decision_simulation_assumption_driven": True,
    }


@router.get("/model-summary")
def model_summary(request: Request) -> dict[str, Any]:
    return load_model_summary(request.app.state.repo_root)


@router.get("/model-comparison")
def model_comparison(request: Request) -> dict[str, Any]:
    df = load_model_comparison(request.app.state.repo_root)
    return {"rows": serialize_records(df)}


@router.get("/feature-importance")
def feature_importance(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    df = load_feature_importance(request.app.state.repo_root)
    ordered = df.sort_values("importance", ascending=False)
    return {
        "total_rows": int(len(ordered)),
        "returned_rows": int(min(len(ordered), limit)),
        "rows": serialize_records(ordered.head(limit)),
    }
