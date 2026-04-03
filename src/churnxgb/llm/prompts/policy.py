from __future__ import annotations

from typing import Any

from churnxgb.llm.prompts.base import build_action_messages


def build_policy_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Compare the selected policy against the baseline using the supplied metrics, overlap, selected budget, and caveats. "
        "Focus on business tradeoffs and whether the recommended policy is materially better."
    )
    return build_action_messages("explain_policy", guidance, context)


def build_budget_tradeoff_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Explain the selected budget tradeoff using the budget frontier, selected budget, baseline metrics, and caveats. "
        "Describe what is gained and what is sacrificed by moving to this budget."
    )
    return build_action_messages("explain_budget_tradeoff", guidance, context)
