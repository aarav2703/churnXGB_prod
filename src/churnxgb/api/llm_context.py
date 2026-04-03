from __future__ import annotations

from typing import Any

from churnxgb.api.artifacts import (
    filter_saved_prediction_row,
    load_backtest_detail,
    load_budget_frontier,
    load_decision_drift,
    load_model_summary,
    load_saved_scored_predictions,
    load_segment_evaluation,
)
from churnxgb.api.serializers import serialize_prediction_output
from churnxgb.modeling.interpretability import explain_prediction_rows
from churnxgb.pipeline.score import simulate_policy_by_budget


DEFAULT_ASSUMPTION_FLAGS = [
    "Decision economics are assumption-driven and do not estimate causal uplift.",
]


def _base_context(payload) -> dict[str, Any]:
    return {
        "page": payload.page,
        "selected_budget": payload.selected_budget,
        "selected_policy": payload.selected_policy,
        "selected_model": payload.selected_model,
        "selected_segment": payload.selected_segment,
        "selected_customer": payload.selected_customer,
        "chart_data": payload.chart_data,
        "key_metrics": dict(payload.key_metrics),
        "baseline_metrics": dict(payload.baseline_metrics),
        "caveats": list(payload.caveats),
        "assumption_flags": list(payload.assumption_flags) or list(DEFAULT_ASSUMPTION_FLAGS),
    }


def build_customer_context(repo_root, model_info: dict[str, Any], payload) -> dict[str, Any]:
    scored = load_saved_scored_predictions(repo_root)
    row = filter_saved_prediction_row(scored, payload.customer_id, payload.invoice_month)
    explanations = explain_prediction_rows(
        model=model_info["model"],
        X=row[model_info["feature_cols"]],
        feature_cols=model_info["feature_cols"],
        top_n=payload.top_n,
    )
    prediction = serialize_prediction_output(row)[0]
    prediction_payload = {
        "customer_id": prediction.get("CustomerID"),
        "invoice_month": prediction.get("invoice_month"),
        "churn_prob": prediction.get("churn_prob"),
        "value_pos": prediction.get("value_pos"),
        "policy_ml": prediction.get("policy_ml"),
        "policy_net_benefit": prediction.get("policy_net_benefit"),
    }
    recommended_action = "review_before_contact"
    if prediction.get("policy_net_benefit") is not None and float(prediction["policy_net_benefit"]) > 0:
        recommended_action = "prioritize_retention_outreach"
    elif prediction.get("churn_prob") is not None and float(prediction["churn_prob"]) >= 0.5:
        recommended_action = "high_risk_but_low_economic_return"

    context = _base_context(payload)
    context["selected_customer"] = {
        "customer_id": str(payload.customer_id),
        "invoice_month": str(payload.invoice_month),
    }
    context["key_metrics"] = {
        **prediction_payload,
        **context["key_metrics"],
    }
    context["customer_context"] = {
        "prediction": prediction_payload,
        "recommended_action": recommended_action,
        **explanations[0],
    }
    return context


def build_policy_context(repo_root, payload) -> dict[str, Any]:
    scored = load_saved_scored_predictions(repo_root)
    selected_budget = payload.selected_budget or 10
    simulation = simulate_policy_by_budget(
        scored,
        [float(selected_budget) / 100.0],
        baseline_policy=payload.baseline_policy,
        comparison_policy=payload.comparison_policy,
    )[0]
    context = _base_context(payload)
    context["selected_budget"] = selected_budget
    context["selected_policy"] = payload.comparison_policy
    context["baseline_metrics"] = {
        **context["baseline_metrics"],
        "baseline_policy": payload.baseline_policy,
        "value_at_risk": simulation.get("value_at_risk_baseline"),
        "net_benefit_at_k": simulation.get("baseline_net_benefit_at_k"),
    }
    context["key_metrics"] = {
        **context["key_metrics"],
        "comparison_policy": payload.comparison_policy,
        "value_at_risk": simulation.get("value_at_risk_comparison"),
        "net_benefit_at_k": simulation.get("comparison_net_benefit_at_k"),
        "comparison_minus_baseline": simulation.get("comparison_minus_baseline"),
        "selection_overlap_at_k": simulation.get("selection_overlap_at_k"),
    }
    context["policy_comparison"] = simulation
    return context


def build_chart_context(repo_root, payload) -> dict[str, Any]:
    context = _base_context(payload)
    chart_type = payload.chart_type
    context["chart_type"] = chart_type
    context["selected_point"] = payload.selected_point
    selected_model = payload.selected_model or load_model_summary(repo_root)["manifest"].get("best_model")
    context["selected_model"] = selected_model

    if chart_type == "budget_frontier":
        frontier = load_budget_frontier(repo_root, str(selected_model))
        context["chart_data"] = frontier.to_dict(orient="records")
        selected_budget = payload.selected_budget or 10
        frontier_row = frontier[frontier["budget_k"] == float(selected_budget) / 100.0]
        if len(frontier_row):
            context["selected_point"] = frontier_row.iloc[0].to_dict()
            context["key_metrics"] = {**context["key_metrics"], **frontier_row.iloc[0].to_dict()}
    elif chart_type == "backtest_stability":
        backtest = load_backtest_detail(repo_root)
        context["chart_data"] = backtest.to_dict(orient="records")
        context["key_metrics"] = {
            **context["key_metrics"],
            "rows": int(len(backtest)),
        }
    elif chart_type == "decision_drift":
        drift = load_decision_drift(repo_root)
        context["chart_data"] = drift.to_dict(orient="records")
        context["key_metrics"] = {
            **context["key_metrics"],
            "rows": int(len(drift)),
        }
    return context


def build_segment_context(repo_root, payload) -> dict[str, Any]:
    segments = load_segment_evaluation(repo_root)
    use = segments[
        (segments["split"] == payload.split)
        & (segments["segment_type"] == payload.segment_type)
        & (segments["segment_value"] == payload.segment_value)
    ].copy()
    context = _base_context(payload)
    context["selected_segment"] = {
        "segment_type": payload.segment_type,
        "segment_value": payload.segment_value,
        "split": payload.split,
    }
    if len(use):
        row = use.iloc[0].to_dict()
        context["key_metrics"] = {**context["key_metrics"], **row}
        context["chart_data"] = use.to_dict(orient="records")
    return context


def build_budget_tradeoff_context(repo_root, payload) -> dict[str, Any]:
    summary = load_model_summary(repo_root)
    selected_model = payload.selected_model or summary["manifest"].get("best_model")
    frontier = load_budget_frontier(repo_root, str(selected_model))
    selected_budget = payload.selected_budget or 10
    row = frontier[frontier["budget_k"] == float(selected_budget) / 100.0]
    context = _base_context(payload)
    context["selected_model"] = selected_model
    context["chart_data"] = frontier.to_dict(orient="records")
    if len(row):
        context["key_metrics"] = {**context["key_metrics"], **row.iloc[0].to_dict()}
        context["selected_point"] = row.iloc[0].to_dict()
    return context


def build_recommendation_context(repo_root, payload) -> dict[str, Any]:
    context = build_budget_tradeoff_context(repo_root, payload)
    context["recommendation_summary"] = {
        "selected_budget": context.get("selected_budget"),
        "selected_policy": context.get("selected_policy"),
        "selected_model": context.get("selected_model"),
    }
    return context


def build_risk_context(repo_root, payload) -> dict[str, Any]:
    context = build_budget_tradeoff_context(repo_root, payload)
    drift = load_decision_drift(repo_root)
    backtest = load_backtest_detail(repo_root)
    context["risk_context"] = {
        "decision_drift_rows": int(len(drift)),
        "backtest_rows": int(len(backtest)),
    }
    context["key_metrics"] = {
        **context["key_metrics"],
        **context["risk_context"],
    }
    return context
