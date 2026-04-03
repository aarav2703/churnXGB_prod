from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


ACTION_SYSTEM_PROMPT = """You are the ChurnXGB grounded explanation engine.

Rules:
- Use only the structured context provided in the user message.
- Never invent metrics, identifiers, rankings, customers, dates, or model behavior.
- Respect assumption-driven caveats exactly when present.
- Return valid JSON with exactly these keys:
  what_this_shows
  why_it_matters
  what_to_do
  caution
- Keep each field concise and decision-focused.
"""


@dataclass
class ExplanationSections:
    what_this_shows: str
    why_it_matters: str
    what_to_do: str
    caution: str

    def as_dict(self) -> dict[str, str]:
        return {
            "what_this_shows": self.what_this_shows,
            "why_it_matters": self.why_it_matters,
            "what_to_do": self.what_to_do,
            "caution": self.caution,
        }

    def as_markdown(self) -> str:
        return (
            f"1. What this shows: {self.what_this_shows}\n"
            f"2. Why it matters: {self.why_it_matters}\n"
            f"3. What to do: {self.what_to_do}\n"
            f"4. Caution: {self.caution}"
        )


def build_action_messages(
    action_name: str,
    guidance: str,
    context: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "action": action_name,
        "guidance": guidance,
        "context": context,
    }
    return [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate a grounded explanation for this decision-analytics action.\n\n"
                f"{json.dumps(payload, indent=2, default=str)}"
            ),
        },
    ]


def parse_sections(payload: dict[str, Any]) -> ExplanationSections:
    return ExplanationSections(
        what_this_shows=str(payload.get("what_this_shows", "")).strip(),
        why_it_matters=str(payload.get("why_it_matters", "")).strip(),
        what_to_do=str(payload.get("what_to_do", "")).strip(),
        caution=str(payload.get("caution", "")).strip(),
    )
