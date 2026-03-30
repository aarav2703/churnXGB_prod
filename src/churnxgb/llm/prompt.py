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
