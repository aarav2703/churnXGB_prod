from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any
from urllib import error, request

from churnxgb.llm.prompt import build_summary_messages
from churnxgb.llm.tools import ChurnToolExecutor


logger = logging.getLogger(__name__)


@dataclass
class LLMToolAnswer:
    answer: str
    tools_used: list[dict[str, Any]]
    raw_data: list[dict[str, Any]] | None = None


class ChurnLLMAgent:
    def __init__(
        self,
        tool_executor: ChurnToolExecutor,
        llm_config: dict[str, Any] | None = None,
    ):
        self.tool_executor = tool_executor
        self.llm_config = llm_config or {}

    def answer_query(self, query: str, include_raw_data: bool = False) -> LLMToolAnswer:
        tool_calls = self._route_query(query)
        tool_results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            logger.info("llm_tool_call name=%s params=%s", tool_call["name"], tool_call["params"])
            tool_results.append(
                self.tool_executor.execute(tool_call["name"], tool_call["params"])
            )

        answer = self._summarize(query, tool_calls, tool_results)
        return LLMToolAnswer(
            answer=answer,
            tools_used=tool_calls,
            raw_data=tool_results if include_raw_data else None,
        )

    def _route_query(self, query: str) -> list[dict[str, Any]]:
        q = query.lower()
        budget_pct = self._extract_budget_pct(query)

        if any(term in q for term in ["drift", "changed recently", "data changed", "monitoring"]):
            return [{"name": "get_drift_summary", "params": {"limit": 10}}]

        if any(term in q for term in ["why is this customer", "explain customer", "high risk", "why high risk"]):
            customer_id = self._extract_customer_id(query)
            invoice_month = self._extract_invoice_month(query)
            params: dict[str, Any] = {"top_n": 5}
            if customer_id and invoice_month:
                params.update({"customer_id": customer_id, "invoice_month": invoice_month})
            return [{"name": "explain_customer", "params": params}]

        if any(term in q for term in ["experiment", "uplift", "treatment", "control", "baseline better"]):
            budgets = [budget_pct / 100.0] if budget_pct is not None else None
            return [{"name": "simulate_experiment", "params": {"budgets": budgets}}]

        if any(term in q for term in ["target", "who should i target", "top customers"]):
            use_budget = budget_pct if budget_pct is not None else 10
            return [{"name": "get_targets", "params": {"budget_pct": use_budget, "limit": 20}}]

        if any(term in q for term in ["simulate", "strategy", "budget", "increase budget", "net benefit", "policy"]):
            budgets = [budget_pct / 100.0] if budget_pct is not None else None
            return [{"name": "simulate_policy", "params": {"budgets": budgets}}]

        return [{"name": "get_model_summary", "params": {}}]

    @staticmethod
    def _extract_budget_pct(query: str) -> int | None:
        match = re.search(r"(\d{1,2})\s*%", query)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_customer_id(query: str) -> str | None:
        match = re.search(r"customer(?:id)?\s*[:=#]?\s*(\d+)", query, flags=re.IGNORECASE)
        return match.group(1) if match else None

    @staticmethod
    def _extract_invoice_month(query: str) -> str | None:
        match = re.search(r"(20\d{2}-\d{2})", query)
        return match.group(1) if match else None

    def _summarize(
        self,
        query: str,
        tools_used: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> str:
        if all(not result.get("ok", False) for result in tool_results):
            errors = "; ".join(result.get("error", "unknown error") for result in tool_results)
            return f"I could not answer from the available backend tools: {errors}"

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            return self._fallback_summary(query, tool_results)

        messages = build_summary_messages(query, tools_used, tool_results)
        return self._call_deepseek(messages, api_key)

    def _call_deepseek(self, messages: list[dict[str, str]], api_key: str) -> str:
        api_base = self.llm_config.get("api_base", "https://api.deepseek.com/v1").rstrip("/")
        model = self.llm_config.get("model", "deepseek-chat")
        timeout_seconds = float(self.llm_config.get("timeout_seconds", 30))
        payload = {
            "model": model,
            "temperature": 0.0,
            "messages": messages,
        }
        req = request.Request(
            f"{api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover - live-network behavior
            detail = exc.read().decode("utf-8", errors="replace")
            return f"I could not get an LLM summary because the DeepSeek API returned {exc.code}: {detail}"
        except Exception as exc:  # pragma: no cover - live-network behavior
            return f"I could not get an LLM summary because the DeepSeek call failed: {exc}"

        try:
            return body["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_summary("tool_summary", [])

    def _fallback_summary(self, query: str, tool_results: list[dict[str, Any]]) -> str:
        first_ok = next((result for result in tool_results if result.get("ok")), None)
        if first_ok is None:
            return "I could not answer from the available backend tools."

        tool_name = first_ok["tool"]
        data = first_ok["data"]

        if tool_name == "get_targets":
            rows = data.get("rows", [])
            top = rows[0] if rows else {}
            return (
                f"Using the {data.get('budget_pct')}% target list, I found {data.get('total_rows')} selected customers. "
                f"The first returned customer is {top.get('CustomerID')} with policy_net_benefit {top.get('policy_net_benefit')}."
            )
        if tool_name == "simulate_policy":
            result = (data.get("results") or [{}])[0]
            return (
                f"Policy simulation at budget {result.get('budget_k')} shows selection overlap {result.get('selection_overlap_at_k')} "
                f"and comparison_minus_baseline {result.get('comparison_minus_baseline')}. "
                f"This output is assumption-driven."
            )
        if tool_name == "simulate_experiment":
            result = (data.get("results") or [{}])[0]
            return (
                f"Experiment simulation at budget {result.get('budget_k')} targets {result.get('targeted_customers')} customers "
                f"with incremental_retained_value {result.get('incremental_retained_value')}. "
                f"This is assumption-driven and not causal inference."
            )
        if tool_name == "explain_customer":
            prediction = data.get("prediction", {})
            positives = data.get("top_positive_contributors", [])
            top_feature = positives[0]["feature"] if positives else "n/a"
            return (
                f"Customer {data.get('identifiers', {}).get('CustomerID')} has churn_prob {prediction.get('churn_prob')}. "
                f"The top positive contributor returned by the explanation tool is {top_feature}."
            )
        if tool_name == "get_drift_summary":
            latest = data.get("latest", {})
            summary = latest.get("summary", {})
            return (
                f"The latest drift snapshot reports {summary.get('n_warn')} warnings and {summary.get('n_alert')} alerts. "
                f"Recent drift history rows returned: {data.get('history', {}).get('returned_rows')}."
            )
        if tool_name == "get_model_summary":
            manifest = data.get("manifest", {})
            return (
                f"The current promoted setup reports best_model {manifest.get('best_model')} "
                f"with selection_policy {manifest.get('selection_policy')}."
            )

        return f"I routed your query through {tool_name} and returned the backend data without an external LLM summary."
