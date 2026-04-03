from __future__ import annotations

from typing import Any

from churnxgb.llm.prompts.base import build_action_messages


def build_recommendation_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Produce a short recommendation summary from the supplied selected budget, policy, key metrics, baseline metrics, caveats, and assumption flags. "
        "Be decisive but grounded."
    )
    return build_action_messages("summarize_recommendation", guidance, context)


def build_risk_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Summarize the main risk in the current decision view using the supplied monitoring, stability, baseline comparison, caveats, and assumption flags. "
        "Do not add risks that are not present in the context."
    )
    return build_action_messages("summarize_risk", guidance, context)
