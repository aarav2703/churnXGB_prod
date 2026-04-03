from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]]
    top_n: int = 5


class SimulatePolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budgets: list[float] | None = None


class SimulateExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budgets: list[float] | None = None


class LLMExplainCustomerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    customer_id: str
    invoice_month: str
    top_n: int = 5
    debug: bool = False


class LLMExplainCustomerCompatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str
    invoice_month: str
    top_n: int = 5
    debug: bool = False


class LLMExplainChartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    chart_type: str
    selected_point: dict[str, Any] | None = None
    debug: bool = False


class LLMExplainSegmentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    segment_type: str
    segment_value: str
    split: str = "test"
    debug: bool = False


class LLMExplainPolicyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    baseline_policy: str = "policy_ml"
    comparison_policy: str = "policy_net_benefit"
    debug: bool = False


class LLMExplainBudgetTradeoffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    debug: bool = False


class LLMSummarizeRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    debug: bool = False


class LLMSummarizeRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: str
    selected_budget: int | None
    selected_policy: str | None
    selected_model: str | None
    selected_segment: dict[str, Any] | None = None
    selected_customer: dict[str, Any] | None = None
    chart_data: list[dict[str, Any]] | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_metrics: dict[str, Any] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    assumption_flags: list[str] = Field(default_factory=list)
    debug: bool = False


class LLMQueryCompatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    include_raw_data: bool = False
