from __future__ import annotations

from typing import Any

from churnxgb.llm.prompts.base import build_action_messages


def build_chart_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Explain the selected chart using the supplied page, chart data, selected point, key metrics, baseline metrics, caveats, and assumption flags. "
        "State the main comparison cleanly and keep the recommendation tied to the selected budget and policy."
    )
    return build_action_messages("explain_chart", guidance, context)
