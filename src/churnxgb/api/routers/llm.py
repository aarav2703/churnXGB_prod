from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from churnxgb.api.llm_context import (
    build_budget_tradeoff_context,
    build_chart_context,
    build_customer_context,
    build_policy_context,
    build_recommendation_context,
    build_risk_context,
    build_segment_context,
)
from churnxgb.api.schemas import (
    LLMExplainBudgetTradeoffRequest,
    LLMExplainChartRequest,
    LLMExplainCustomerCompatRequest,
    LLMExplainCustomerRequest,
    LLMExplainPolicyRequest,
    LLMExplainSegmentRequest,
    LLMQueryCompatRequest,
    LLMSummarizeRecommendationRequest,
    LLMSummarizeRiskRequest,
)
from churnxgb.llm.actions import (
    deprecated_query_response,
    explain_budget_tradeoff,
    explain_chart,
    explain_customer,
    explain_policy,
    explain_segment,
    summarize_recommendation,
    summarize_risk,
)


router = APIRouter(tags=["llm"])


@router.post("/llm/explain/customer")
def explain_customer_endpoint(
    request: Request,
    payload: LLMExplainCustomerRequest,
) -> dict[str, Any]:
    context = build_customer_context(
        request.app.state.repo_root,
        request.app.state.model_info,
        payload,
    )
    result = explain_customer(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/explain_customer", deprecated=True)
def explain_customer_compat(
    request: Request,
    payload: LLMExplainCustomerCompatRequest,
) -> dict[str, Any]:
    expanded = LLMExplainCustomerRequest(
        page="Customer Explorer",
        selected_budget=10,
        selected_policy="policy_net_benefit",
        selected_model=request.app.state.model_info.get("model_name"),
        caveats=["Economic outputs are assumption-driven."],
        assumption_flags=["No causal uplift is estimated."],
        customer_id=payload.customer_id,
        invoice_month=payload.invoice_month,
        top_n=payload.top_n,
        debug=payload.debug,
    )
    return explain_customer_endpoint(request, expanded)


@router.post("/llm/explain/chart")
def explain_chart_endpoint(
    request: Request,
    payload: LLMExplainChartRequest,
) -> dict[str, Any]:
    context = build_chart_context(request.app.state.repo_root, payload)
    result = explain_chart(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/explain/segment")
def explain_segment_endpoint(
    request: Request,
    payload: LLMExplainSegmentRequest,
) -> dict[str, Any]:
    context = build_segment_context(request.app.state.repo_root, payload)
    result = explain_segment(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/explain/policy")
def explain_policy_endpoint(
    request: Request,
    payload: LLMExplainPolicyRequest,
) -> dict[str, Any]:
    context = build_policy_context(request.app.state.repo_root, payload)
    result = explain_policy(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/explain/budget-tradeoff")
def explain_budget_tradeoff_endpoint(
    request: Request,
    payload: LLMExplainBudgetTradeoffRequest,
) -> dict[str, Any]:
    context = build_budget_tradeoff_context(request.app.state.repo_root, payload)
    result = explain_budget_tradeoff(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/summarize/recommendation")
def summarize_recommendation_endpoint(
    request: Request,
    payload: LLMSummarizeRecommendationRequest,
) -> dict[str, Any]:
    context = build_recommendation_context(request.app.state.repo_root, payload)
    result = summarize_recommendation(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/summarize/risk")
def summarize_risk_endpoint(
    request: Request,
    payload: LLMSummarizeRiskRequest,
) -> dict[str, Any]:
    context = build_risk_context(request.app.state.repo_root, payload)
    result = summarize_risk(context, request.app.state.llm_config)
    return result.to_response(debug=payload.debug)


@router.post("/llm/query", deprecated=True)
def query_compat(_: Request, __: LLMQueryCompatRequest) -> dict[str, Any]:
    return deprecated_query_response().to_response(debug=False)
