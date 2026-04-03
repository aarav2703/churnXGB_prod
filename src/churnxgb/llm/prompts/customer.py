from __future__ import annotations

from typing import Any

from churnxgb.llm.prompts.base import build_action_messages


def build_customer_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Explain why the selected customer was targeted using only the supplied customer identifiers, churn/value metrics, feature contributions, segment labels, and caveats. "
        "Recommend an action consistent with the provided recommendation field."
    )
    return build_action_messages("explain_customer", guidance, context)
