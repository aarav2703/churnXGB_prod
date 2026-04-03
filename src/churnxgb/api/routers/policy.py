from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from churnxgb.api.artifacts import (
    load_budget_frontier,
    load_model_summary,
    load_policy_metrics,
    load_saved_scored_predictions,
    load_segment_evaluation,
)
from churnxgb.api.schemas import SimulateExperimentRequest, SimulatePolicyRequest
from churnxgb.api.serializers import serialize_records
from churnxgb.evaluation.experiment_simulation import simulate_experiment_by_budget
from churnxgb.pipeline.score import simulate_policy_by_budget


router = APIRouter(tags=["policy"])


@router.get("/policy-metrics")
def policy_metrics(
    request: Request,
    model_name: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    summary = load_model_summary(request.app.state.repo_root)
    manifest = summary["manifest"]
    use_model = model_name or manifest.get("best_model")
    df = load_policy_metrics(request.app.state.repo_root, str(use_model), split)
    return {
        "model_name": use_model,
        "split": split,
        "rows": serialize_records(df),
    }


@router.get("/frontier")
def decision_frontier(
    request: Request,
    model_name: str | None = None,
) -> dict[str, Any]:
    summary = load_model_summary(request.app.state.repo_root)
    manifest = summary["manifest"]
    use_model = model_name or manifest.get("best_model")
    df = load_budget_frontier(request.app.state.repo_root, str(use_model))
    return {"model_name": use_model, "rows": serialize_records(df)}


@router.get("/segments")
def segment_metrics(
    request: Request,
    split: str = "test",
    segment_type: str | None = None,
) -> dict[str, Any]:
    df = load_segment_evaluation(request.app.state.repo_root)
    df = df[df["split"] == split].copy()
    if segment_type is not None:
        df = df[df["segment_type"] == segment_type].copy()
    return {"rows": serialize_records(df)}


@router.post("/simulate-policy")
def simulate_policy(
    request: Request,
    payload: SimulatePolicyRequest,
) -> dict[str, Any]:
    use_budgets = (
        [float(x) for x in payload.budgets]
        if payload.budgets is not None
        else list(request.app.state.budgets)
    )
    scored = load_saved_scored_predictions(request.app.state.repo_root)
    return {
        "assumption_driven": True,
        "note": "This endpoint simulates decision economics from flat configured intervention assumptions on saved offline scored predictions. It is assumption-driven threshold analysis, not causal inference or observed treatment effect estimation.",
        "decision_config": dict(request.app.state.decision_cfg),
        "results": simulate_policy_by_budget(scored, use_budgets),
    }


@router.post("/simulate-experiment")
def simulate_experiment(
    request: Request,
    payload: SimulateExperimentRequest,
) -> dict[str, Any]:
    use_budgets = (
        [float(x) for x in payload.budgets]
        if payload.budgets is not None
        else list(request.app.state.budgets)
    )
    scored = load_saved_scored_predictions(request.app.state.repo_root)
    return {
        "assumption_driven": True,
        "note": "This endpoint runs a deterministic business-case simulation from configured treatment assumptions on saved offline scored predictions. It is not causal inference, not observed uplift estimation, and does not produce experimental confidence intervals.",
        "experiment_config": dict(request.app.state.experiment_cfg),
        "results": simulate_experiment_by_budget(
            scored,
            use_budgets,
            request.app.state.experiment_cfg,
        ),
    }
