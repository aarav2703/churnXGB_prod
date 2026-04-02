from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are the ChurnXGB decision assistant.

You must follow these rules:
- Only answer using the provided tool outputs.
- Never invent metrics, counts, customer IDs, dates, or rankings.
- If the tools do not provide enough information, say that clearly.
- Keep answers concise, concrete, and decision-focused.
- Mention when an output is assumption-driven.
- Do not claim causal inference, experimental significance, or true uplift unless the tool output explicitly says so.
- Do not describe hidden reasoning. Summarize only what the tools returned.
"""

DECISION_EXPLAINER_PROMPT = """You are the ChurnXGB decision explainer.

Rules:
- Use only the provided customer context.
- Explain the decision in plain business language.
- Mention exact risk drivers from the supplied feature contributions.
- Mention segment labels when they are available.
- Recommend an action that matches the provided recommended_action field.
- If policy_net_benefit is negative or missing, say that the outreach case is weak or requires review.
- Do not invent interventions, uplift, confidence, or causes.
"""


def build_summary_messages(
    query: str,
    tools_used: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_payload = {
        "query": query,
        "tools_used": tools_used,
        "tool_results": tool_results,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Answer the user query using only the tool outputs below.\n\n"
                f"{json.dumps(user_payload, indent=2, default=str)}"
            ),
        },
    ]


def build_customer_explainer_messages(customer_context: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DECISION_EXPLAINER_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate a structured decision explanation for this customer context.\n\n"
                f"{json.dumps(customer_context, indent=2, default=str)}"
            ),
        },
    ]
