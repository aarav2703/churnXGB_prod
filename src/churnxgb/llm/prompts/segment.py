from __future__ import annotations

from typing import Any

from churnxgb.llm.prompts.base import build_action_messages


def build_segment_messages(context: dict[str, Any]) -> list[dict[str, str]]:
    guidance = (
        "Summarize the selected segment using only the segment metrics, comparison values, selected budget, and caveats. "
        "Make clear whether the segment looks strong, weak, or negative ROI."
    )
    return build_action_messages("explain_segment", guidance, context)
