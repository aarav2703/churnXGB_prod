from __future__ import annotations

from churnxgb.llm.actions import (
    ExplanationResult,
    deprecated_query_response,
    explain_budget_tradeoff,
    explain_chart,
    explain_customer,
    explain_policy,
    explain_segment,
    summarize_recommendation,
    summarize_risk,
)
from churnxgb.llm.prompts.base import ExplanationSections

__all__ = [
    "ExplanationResult",
    "ExplanationSections",
    "explain_chart",
    "explain_customer",
    "explain_segment",
    "explain_policy",
    "explain_budget_tradeoff",
    "summarize_recommendation",
    "summarize_risk",
    "deprecated_query_response",
]
