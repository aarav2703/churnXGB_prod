from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from churnxgb.evaluation.experiment_simulation import simulate_experiment_by_budget
from churnxgb.inference.contracts import (
    IDENTIFIER_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
    build_prediction_output,
)
from churnxgb.modeling.interpretability import explain_prediction_rows
from churnxgb.monitoring.history import load_drift_history
from churnxgb.paths import resolve_runtime_root
from churnxgb.pipeline.score import score_dataframe, simulate_policy_by_budget


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "get_targets",
        "description": "Fetch the current scored target list for a budget percentage from the saved offline outputs.",
        "endpoint": "GET /targets/{budget_pct}",
        "input_schema": {
            "type": "object",
            "properties": {
                "budget_pct": {"type": "integer", "minimum": 1, "maximum": 100},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["budget_pct"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "budget_pct": {"type": "integer"},
                "total_rows": {"type": "integer"},
                "returned_rows": {"type": "integer"},
                "rows": {"type": "array"},
            },
        },
    },
    {
        "name": "simulate_policy",
        "description": "Compare baseline and cost-aware policy behavior over saved scored outputs.",
        "endpoint": "POST /simulate-policy",
        "input_schema": {
            "type": "object",
            "properties": {
                "budgets": {"type": "array", "items": {"type": "number"}},
            },
        },
        "output_schema": {"type": "object"},
    },
    {
        "name": "simulate_experiment",
        "description": "Run the assumption-driven experiment simulator over saved scored outputs.",
        "endpoint": "POST /simulate-experiment",
        "input_schema": {
            "type": "object",
            "properties": {
                "budgets": {"type": "array", "items": {"type": "number"}},
            },
        },
        "output_schema": {"type": "object"},
    },
    {
        "name": "explain_customer",
        "description": "Explain either a saved scored customer row or an ad hoc prediction payload using the existing explanation API logic.",
        "endpoint": "GET /customers/explain or POST /explain",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "invoice_month": {"type": "string"},
                "rows": {"type": "array"},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 20},
            },
        },
        "output_schema": {"type": "object"},
    },
    {
        "name": "get_drift_summary",
        "description": "Fetch the latest drift summary and the recent drift history rows.",
        "endpoint": "GET /drift/latest and GET /drift/history",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
        },
        "output_schema": {"type": "object"},
    },
    {
        "name": "get_model_summary",
        "description": "Fetch the current model summary, manifest, promotion record, and best-model comparison row.",
        "endpoint": "GET /model-summary",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object"},
    },
]


@dataclass
class ToolContext:
    repo_root: Path
    model_info: dict[str, Any]
    budgets: list[float]
    decision_cfg: dict[str, Any]
    experiment_cfg: dict[str, Any]
    monitoring_cfg: dict[str, Any]


class ToolExecutionError(RuntimeError):
    pass


def _serialize_prediction_output(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = build_prediction_output(df).copy()
    if "invoice_month" in out.columns:
        out["invoice_month"] = out["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    if "T" in out.columns:
        out["T"] = out["T"].map(
            lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
        )
    out = out.where(pd.notna(out), None)
    return out[PREDICTION_OUTPUT_COLUMNS].to_dict(orient="records")


def _serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    if "invoice_month" in out.columns:
        out["invoice_month"] = out["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    if "T" in out.columns:
        out["T"] = out["T"].map(
            lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
        )
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


class ChurnToolExecutor:
    def __init__(self, context: ToolContext):
        self.context = context

    def definitions(self) -> list[dict[str, Any]]:
        return TOOL_DEFINITIONS

    def execute(self, name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = params or {}
        try:
            if name == "get_targets":
                data = self.get_targets(**params)
            elif name == "simulate_policy":
                data = self.simulate_policy(**params)
            elif name == "simulate_experiment":
                data = self.simulate_experiment(**params)
            elif name == "explain_customer":
                data = self.explain_customer(**params)
            elif name == "get_drift_summary":
                data = self.get_drift_summary(**params)
            elif name == "get_model_summary":
                data = self.get_model_summary()
            else:
                raise ToolExecutionError(f"Unknown tool: {name}")
            return {"ok": True, "tool": name, "data": data}
        except Exception as exc:  # pragma: no cover - exercised via endpoint behavior
            return {
                "ok": False,
                "tool": name,
                "error": str(exc),
            }

    def _runtime_root(self) -> Path:
        return resolve_runtime_root(self.context.repo_root)

    def _saved_predictions(self) -> pd.DataFrame:
        path = self._runtime_root() / "outputs" / "predictions" / "predictions_all.parquet"
        if not path.exists():
            raise ToolExecutionError(
                "Saved scored predictions not found. Run the offline scoring pipeline first."
            )
        return pd.read_parquet(path)

    def get_targets(self, budget_pct: int, limit: int = 20) -> dict[str, Any]:
        path = self._runtime_root() / "outputs" / "targets" / f"targets_all_k{int(budget_pct):02d}.parquet"
        if not path.exists():
            raise ToolExecutionError(f"Target list not found for budget {budget_pct}%.")
        df = pd.read_parquet(path)
        return {
            "source_endpoint": f"/targets/{int(budget_pct)}",
            "budget_pct": int(budget_pct),
            "total_rows": int(len(df)),
            "returned_rows": int(min(len(df), limit)),
            "rows": _serialize_records(df.head(limit)),
        }

    def simulate_policy(self, budgets: list[float] | None = None) -> dict[str, Any]:
        scored = self._saved_predictions()
        use_budgets = [float(x) for x in (budgets or self.context.budgets)]
        return {
            "source_endpoint": "/simulate-policy",
            "assumption_driven": True,
            "decision_config": dict(self.context.decision_cfg),
            "results": simulate_policy_by_budget(scored, use_budgets),
        }

    def simulate_experiment(self, budgets: list[float] | None = None) -> dict[str, Any]:
        scored = self._saved_predictions()
        use_budgets = [float(x) for x in (budgets or self.context.budgets)]
        return {
            "source_endpoint": "/simulate-experiment",
            "assumption_driven": True,
            "experiment_config": dict(self.context.experiment_cfg),
            "results": simulate_experiment_by_budget(
                scored,
                use_budgets,
                self.context.experiment_cfg,
            ),
        }

    def explain_customer(
        self,
        customer_id: str | None = None,
        invoice_month: str | None = None,
        rows: list[dict[str, Any]] | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        model_info = self.context.model_info
        if rows:
            request_df = pd.DataFrame(rows)
            scored = score_dataframe(
                request_df,
                model=model_info["model"],
                feature_cols=model_info["feature_cols"],
                contract=model_info["contract"],
                budgets=self.context.budgets,
                model_source=model_info["model_source"],
                decision_cfg=self.context.decision_cfg,
            )
            explanations = explain_prediction_rows(
                model=model_info["model"],
                X=request_df[model_info["feature_cols"]],
                feature_cols=model_info["feature_cols"],
                top_n=top_n,
            )
            prediction = _serialize_prediction_output(scored)[0]
            return {
                "source_endpoint": "/explain",
                "identifiers": {
                    key: prediction.get(key) for key in IDENTIFIER_COLUMNS
                },
                "prediction": {
                    key: prediction.get(key)
                    for key in PREDICTION_OUTPUT_COLUMNS
                    if key not in IDENTIFIER_COLUMNS
                },
                **explanations[0],
            }

        if customer_id is None or invoice_month is None:
            raise ToolExecutionError(
                "explain_customer requires either rows or both customer_id and invoice_month."
            )

        saved = self._saved_predictions().copy()
        if "invoice_month" in saved.columns:
            saved["invoice_month"] = saved["invoice_month"].map(
                lambda x: None if pd.isna(x) else str(x)
            )
        matched = saved[
            (saved["CustomerID"].astype(str) == str(customer_id))
            & (saved["invoice_month"] == str(invoice_month))
        ].head(1)
        if len(matched) == 0:
            raise ToolExecutionError(
                f"No saved scored row found for CustomerID={customer_id}, invoice_month={invoice_month}."
            )

        explanations = explain_prediction_rows(
            model=model_info["model"],
            X=matched[model_info["feature_cols"]],
            feature_cols=model_info["feature_cols"],
            top_n=top_n,
        )
        prediction = _serialize_prediction_output(matched)[0]
        return {
            "source_endpoint": "/customers/explain",
            "identifiers": {
                key: prediction.get(key) for key in IDENTIFIER_COLUMNS
            },
            "prediction": {
                key: prediction.get(key)
                for key in PREDICTION_OUTPUT_COLUMNS
                if key not in IDENTIFIER_COLUMNS
            },
            **explanations[0],
        }

    def get_drift_summary(self, limit: int = 10) -> dict[str, Any]:
        runtime_root = self._runtime_root()
        latest_path = runtime_root / "reports" / "monitoring" / "drift_latest.json"
        history_path = runtime_root / "reports" / "monitoring" / "drift_history.csv"
        if not latest_path.exists():
            raise ToolExecutionError("drift_latest.json not found.")
        with open(latest_path, "r", encoding="utf-8") as f:
            latest = json.load(f)
        history = load_drift_history(history_path)
        history_rows = []
        if len(history):
            history_rows = _serialize_records(
                history.sort_values("generated_at_utc", ascending=False).head(limit)
            )
        return {
            "source_endpoint": "/drift/latest + /drift/history",
            "monitoring_config": dict(self.context.monitoring_cfg),
            "latest": latest,
            "history": {
                "total_rows": int(len(history)),
                "returned_rows": int(min(len(history), limit)),
                "rows": history_rows,
            },
        }

    def get_model_summary(self) -> dict[str, Any]:
        runtime_root = self._runtime_root()
        manifest_path = runtime_root / "reports" / "training_manifest.json"
        comparison_path = runtime_root / "reports" / "model_comparison.csv"
        promotion_path = runtime_root / "models" / "promoted" / "production.json"
        if not manifest_path.exists():
            raise ToolExecutionError("training_manifest.json not found.")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        with open(promotion_path, "r", encoding="utf-8") as f:
            promotion = json.load(f)
        comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
        comparison_row = None
        best_model = manifest.get("best_model")
        if best_model is not None and len(comparison):
            matched = comparison[comparison["model"] == best_model]
            if len(matched):
                comparison_row = matched.iloc[0].to_dict()
        return {
            "source_endpoint": "/model-summary",
            "manifest": manifest,
            "promotion": promotion,
            "comparison_row": comparison_row,
        }
