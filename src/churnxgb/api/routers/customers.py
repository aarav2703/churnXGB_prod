from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from churnxgb.api.artifacts import (
    filter_saved_prediction_row,
    load_saved_scored_predictions,
    load_target_records,
)
from churnxgb.api.schemas import ExplainRequest, PredictRequest
from churnxgb.api.serializers import (
    customer_prediction_payload,
    prepare_request_frame,
    serialize_prediction_output,
    serialize_records,
)
from churnxgb.modeling.interpretability import explain_prediction_rows
from churnxgb.pipeline.score import score_dataframe


router = APIRouter(tags=["customers"])


@router.get("/targets/{budget_pct}")
def targets(
    request: Request,
    budget_pct: int,
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    df = load_target_records(request.app.state.repo_root, budget_pct)
    return {
        "budget_pct": int(budget_pct),
        "total_rows": int(len(df)),
        "returned_rows": int(min(len(df), limit)),
        "rows": serialize_records(df.head(limit)),
    }


@router.get("/predictions")
def predictions(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
    sort_by: str = Query(default="policy_net_benefit"),
) -> dict[str, Any]:
    df = load_saved_scored_predictions(request.app.state.repo_root)
    if sort_by not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"sort_by column not found: {sort_by}",
        )
    ordered = df.sort_values(sort_by, ascending=False)
    return {
        "sort_by": sort_by,
        "total_rows": int(len(ordered)),
        "returned_rows": int(min(len(ordered), limit)),
        "rows": serialize_records(ordered.head(limit)),
    }


@router.get("/customers/explain")
def explain_saved_customer(
    request: Request,
    customer_id: str,
    invoice_month: str,
    top_n: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    scored = load_saved_scored_predictions(request.app.state.repo_root)
    row = filter_saved_prediction_row(scored, customer_id, invoice_month)
    model_info = request.app.state.model_info
    explanations = explain_prediction_rows(
        model=model_info["model"],
        X=row[model_info["feature_cols"]],
        feature_cols=model_info["feature_cols"],
        top_n=top_n,
    )
    prediction = serialize_prediction_output(row)[0]
    return {
        **customer_prediction_payload(prediction),
        **explanations[0],
    }


@router.post("/predict")
def predict(request: Request, payload: PredictRequest) -> list[dict[str, Any]]:
    model_info = request.app.state.model_info
    request_df = prepare_request_frame(payload.rows, model_info["contract"])
    scored = score_dataframe(
        request_df,
        model=model_info["model"],
        feature_cols=model_info["feature_cols"],
        contract=model_info["contract"],
        budgets=request.app.state.budgets,
        model_source=model_info["model_source"],
        decision_cfg=request.app.state.decision_cfg,
    )
    return serialize_prediction_output(scored)


@router.post("/explain")
def explain(request: Request, payload: ExplainRequest) -> list[dict[str, Any]]:
    model_info = request.app.state.model_info
    request_df = prepare_request_frame(payload.rows, model_info["contract"])
    scored = score_dataframe(
        request_df,
        model=model_info["model"],
        feature_cols=model_info["feature_cols"],
        contract=model_info["contract"],
        budgets=request.app.state.budgets,
        model_source=model_info["model_source"],
        decision_cfg=request.app.state.decision_cfg,
    )
    explanations = explain_prediction_rows(
        model=model_info["model"],
        X=request_df[model_info["feature_cols"]],
        feature_cols=model_info["feature_cols"],
        top_n=payload.top_n,
    )

    rows: list[dict[str, Any]] = []
    prediction_records = serialize_prediction_output(scored)
    for idx, explanation in enumerate(explanations):
        rows.append(
            {
                **customer_prediction_payload(prediction_records[idx]),
                **explanation,
            }
        )

    return rows
