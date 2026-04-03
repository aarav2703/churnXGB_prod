from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable
from urllib import error, request

from churnxgb.llm.prompts.base import ExplanationSections, parse_sections
from churnxgb.llm.prompts.chart import build_chart_messages
from churnxgb.llm.prompts.customer import build_customer_messages
from churnxgb.llm.prompts.policy import (
    build_budget_tradeoff_messages,
    build_policy_messages,
)
from churnxgb.llm.prompts.segment import build_segment_messages
from churnxgb.llm.prompts.summary import (
    build_recommendation_messages,
    build_risk_messages,
)


@dataclass
class ExplanationResult:
    action: str
    sections: ExplanationSections
    context: dict[str, Any] | None = None
    deprecated: bool = False

    def to_response(self, debug: bool = False) -> dict[str, Any]:
        payload = {
            "action": self.action,
            "sections": self.sections.as_dict(),
            "answer": self.sections.as_markdown(),
        }
        if self.deprecated:
            payload["deprecated"] = True
        if debug:
            payload["context"] = self.context
        return payload


PromptBuilder = Callable[[dict[str, Any]], list[dict[str, str]]]
FallbackBuilder = Callable[[dict[str, Any]], ExplanationSections]


def explain_chart(context: dict[str, Any], llm_config: dict[str, Any]) -> ExplanationResult:
    return _run_action(
        action="explain_chart",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_chart_messages,
        fallback_builder=_fallback_chart_sections,
    )


def explain_customer(context: dict[str, Any], llm_config: dict[str, Any]) -> ExplanationResult:
    return _run_action(
        action="explain_customer",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_customer_messages,
        fallback_builder=_fallback_customer_sections,
    )


def explain_segment(context: dict[str, Any], llm_config: dict[str, Any]) -> ExplanationResult:
    return _run_action(
        action="explain_segment",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_segment_messages,
        fallback_builder=_fallback_segment_sections,
    )


def explain_policy(context: dict[str, Any], llm_config: dict[str, Any]) -> ExplanationResult:
    return _run_action(
        action="explain_policy",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_policy_messages,
        fallback_builder=_fallback_policy_sections,
    )


def explain_budget_tradeoff(
    context: dict[str, Any], llm_config: dict[str, Any]
) -> ExplanationResult:
    return _run_action(
        action="explain_budget_tradeoff",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_budget_tradeoff_messages,
        fallback_builder=_fallback_budget_tradeoff_sections,
    )


def summarize_recommendation(
    context: dict[str, Any], llm_config: dict[str, Any]
) -> ExplanationResult:
    return _run_action(
        action="summarize_recommendation",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_recommendation_messages,
        fallback_builder=_fallback_recommendation_sections,
    )


def summarize_risk(context: dict[str, Any], llm_config: dict[str, Any]) -> ExplanationResult:
    return _run_action(
        action="summarize_risk",
        context=context,
        llm_config=llm_config,
        prompt_builder=build_risk_messages,
        fallback_builder=_fallback_risk_sections,
    )


def deprecated_query_response() -> ExplanationResult:
    sections = ExplanationSections(
        what_this_shows="The generic query endpoint has been deprecated.",
        why_it_matters="The LLM layer now expects a structured explanation action tied to dashboard context.",
        what_to_do="Call one of the structured endpoints such as /llm/explain/chart, /llm/explain/customer, /llm/explain/segment, or /llm/explain/policy.",
        caution="Free-form chat routing is no longer supported because it weakens grounding and makes explanations less reliable.",
    )
    return ExplanationResult(
        action="deprecated_query",
        sections=sections,
        context=None,
        deprecated=True,
    )


def _run_action(
    action: str,
    context: dict[str, Any],
    llm_config: dict[str, Any],
    prompt_builder: PromptBuilder,
    fallback_builder: FallbackBuilder,
) -> ExplanationResult:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    sections = fallback_builder(context)
    if api_key:
        messages = prompt_builder(context)
        remote_sections = _call_llm(messages, api_key, llm_config)
        if remote_sections is not None:
            sections = remote_sections
    return ExplanationResult(action=action, sections=sections, context=context)


def _call_llm(
    messages: list[dict[str, str]],
    api_key: str,
    llm_config: dict[str, Any],
) -> ExplanationSections | None:
    api_base = llm_config.get("api_base", "https://api.deepseek.com/v1").rstrip("/")
    model = llm_config.get("model", "deepseek-chat")
    timeout_seconds = float(llm_config.get("timeout_seconds", 30))
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": messages,
        "response_format": {"type": "json_object"},
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
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parse_sections(parsed)
    except (error.HTTPError, error.URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        return None
    except Exception:
        return None


def _fmt_number(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    return f"{number:.{digits}f}%"


def _join_caveats(context: dict[str, Any]) -> str:
    caveats = list(context.get("caveats", [])) + list(context.get("assumption_flags", []))
    return "; ".join([str(item) for item in caveats if item]) or "Use the stated assumptions and saved artifacts only."


def _fallback_chart_sections(context: dict[str, Any]) -> ExplanationSections:
    point = context.get("selected_point") or {}
    selected_budget = context.get("selected_budget")
    return ExplanationSections(
        what_this_shows=(
            f"This {context.get('chart_type', 'chart')} on {context.get('page', 'the current page')} is centered on the {selected_budget}% budget view "
            f"with selected metrics {context.get('key_metrics', {})}."
        ),
        why_it_matters=(
            f"It shows how the current policy {context.get('selected_policy')} is performing relative to the supplied baseline metrics {context.get('baseline_metrics', {})}."
        ),
        what_to_do=(
            f"Use the selected point {point or 'for the current budget'} to decide whether to keep the current budget and policy or move to a nearby point with a better tradeoff."
        ),
        caution=_join_caveats(context),
    )


def _fallback_customer_sections(context: dict[str, Any]) -> ExplanationSections:
    prediction = context.get("key_metrics", {})
    customer = context.get("selected_customer", {})
    top_drivers = context.get("customer_context", {}).get("top_positive_contributors", [])
    top_features = ", ".join(str(row.get("feature")) for row in top_drivers[:3]) or "the strongest available drivers"
    return ExplanationSections(
        what_this_shows=(
            f"Customer {customer.get('customer_id')} in {customer.get('invoice_month')} is scored with churn probability {_fmt_pct(prediction.get('churn_prob', 0) * 100 if isinstance(prediction.get('churn_prob'), (int, float)) and prediction.get('churn_prob', 0) <= 1 else prediction.get('churn_prob'), 1)} "
            f"and policy net benefit {_fmt_number(prediction.get('policy_net_benefit'))}."
        ),
        why_it_matters=(
            f"The main positive drivers in the current explanation are {top_features}, which are pushing this customer toward the targeted set."
        ),
        what_to_do=str(context.get("customer_context", {}).get("recommended_action", "Review whether this customer should stay in the current target list.")),
        caution=_join_caveats(context),
    )


def _fallback_segment_sections(context: dict[str, Any]) -> ExplanationSections:
    segment = context.get("selected_segment", {})
    metrics = context.get("key_metrics", {})
    return ExplanationSections(
        what_this_shows=(
            f"Segment {segment.get('segment_value')} in {segment.get('segment_type')} is being evaluated at budget {context.get('selected_budget')}% with metrics {metrics}."
        ),
        why_it_matters=(
            "It shows whether this segment contributes enough value or net benefit to justify inclusion in the targeting policy."
        ),
        what_to_do=(
            "Prioritize segments with positive economics and strong captured value, and scrutinize segments with weak or negative net benefit."
        ),
        caution=_join_caveats(context),
    )


def _fallback_policy_sections(context: dict[str, Any]) -> ExplanationSections:
    comparison = context.get("key_metrics", {})
    return ExplanationSections(
        what_this_shows=(
            f"The selected policy {context.get('selected_policy')} is being compared against baseline metrics {context.get('baseline_metrics', {})} at budget {context.get('selected_budget')}%."
        ),
        why_it_matters=(
            f"The current comparison metrics {comparison} indicate whether the chosen policy is materially improving value capture or net benefit."
        ),
        what_to_do=(
            "Keep the comparison policy only if it improves the business metric you care about at the selected budget; otherwise stay with the baseline."
        ),
        caution=_join_caveats(context),
    )


def _fallback_budget_tradeoff_sections(context: dict[str, Any]) -> ExplanationSections:
    return ExplanationSections(
        what_this_shows=(
            f"The budget tradeoff view is focused on the {context.get('selected_budget')}% budget point with key metrics {context.get('key_metrics', {})}."
        ),
        why_it_matters=(
            "Changing the budget changes both targeting volume and the economics of the selected set, so the recommendation depends on the frontier shape rather than a single KPI."
        ),
        what_to_do=(
            "Use this view to choose the lowest budget that still gives acceptable net benefit and value capture for the current policy."
        ),
        caution=_join_caveats(context),
    )


def _fallback_recommendation_sections(context: dict[str, Any]) -> ExplanationSections:
    return ExplanationSections(
        what_this_shows=(
            f"The current recommendation context combines budget {context.get('selected_budget')}%, policy {context.get('selected_policy')}, model {context.get('selected_model')}, and key metrics {context.get('key_metrics', {})}."
        ),
        why_it_matters=(
            "This is the shortest decision summary for the current dashboard state and should align the viewer on the main action."
        ),
        what_to_do=(
            "Use the selected budget and policy as the default operating point unless a nearby alternative offers a clearly better tradeoff."
        ),
        caution=_join_caveats(context),
    )


def _fallback_risk_sections(context: dict[str, Any]) -> ExplanationSections:
    return ExplanationSections(
        what_this_shows=(
            f"The current risk summary is based on key metrics {context.get('key_metrics', {})}, baseline metrics {context.get('baseline_metrics', {})}, and caveats {context.get('caveats', [])}."
        ),
        why_it_matters=(
            "A good-looking recommendation can still be fragile if it depends heavily on assumptions, unstable backtests, or weak segment economics."
        ),
        what_to_do=(
            "Treat the main risk as the first thing to monitor or stress-test before relying on the current recommendation."
        ),
        caution=_join_caveats(context),
    )
