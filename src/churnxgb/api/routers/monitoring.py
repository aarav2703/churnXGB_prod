from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from churnxgb.api.artifacts import (
    load_backtest_detail,
    load_decision_drift,
    load_drift_history_records,
    load_latest_drift,
)
from churnxgb.api.serializers import serialize_records


router = APIRouter(tags=["monitoring"])


@router.get("/drift/latest")
def drift_latest(request: Request) -> dict[str, Any]:
    return load_latest_drift(request.app.state.repo_root)


@router.get("/drift/history")
def drift_history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=1000),
) -> dict[str, Any]:
    out = load_drift_history_records(request.app.state.repo_root, limit)
    return {
        "monitoring_config": dict(request.app.state.monitoring_cfg),
        **out,
    }


@router.get("/drift/decision")
def decision_drift(
    request: Request,
    budget_pct: int | None = None,
) -> dict[str, Any]:
    df = load_decision_drift(request.app.state.repo_root)
    if budget_pct is not None:
        df = df[df["budget_k"] == float(budget_pct) / 100.0].copy()
    return {"rows": serialize_records(df.sort_values(["invoice_month", "budget_k"]))}


@router.get("/backtest")
def backtest(
    request: Request,
    model_name: str | None = None,
    budget_pct: int | None = None,
) -> dict[str, Any]:
    df = load_backtest_detail(request.app.state.repo_root)
    if model_name is not None:
        df = df[df["model"] == model_name].copy()
    if budget_pct is not None:
        df = df[df["budget_k"] == float(budget_pct) / 100.0].copy()
    return {"rows": serialize_records(df.sort_values(["fold", "model"]))}
