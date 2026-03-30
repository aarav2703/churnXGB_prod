from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from churnxgb.api.api_llm import create_llm_router
from churnxgb.inference.contracts import (
    IDENTIFIER_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
    TRAINING_ONLY_COLUMNS,
    build_prediction_output,
)
from churnxgb.evaluation.experiment_simulation import (
    get_experiment_config,
    simulate_experiment_by_budget,
)
from churnxgb.llm.agent import ChurnLLMAgent
from churnxgb.llm.tools import ChurnToolExecutor, ToolContext
from churnxgb.modeling.interpretability import explain_prediction_rows
from churnxgb.monitoring.history import load_drift_history
from churnxgb.monitoring.alerts import get_monitoring_alert_config
from churnxgb.paths import resolve_runtime_root
from churnxgb.pipeline.score import load_model, score_dataframe, simulate_policy_by_budget
from churnxgb.policy.scoring import get_decision_policy_config


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


def _repo_root_from_app_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_api_config(repo_root: Path) -> dict:
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {
            "eval": {"budgets": []},
            "mlflow": {"tracking_uri": "file:./mlruns_store"},
            "decision": get_decision_policy_config({}),
            "experiment": get_experiment_config({}),
            "monitoring": get_monitoring_alert_config({}),
        }

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _prepare_request_frame(rows: list[dict[str, Any]], contract: dict) -> pd.DataFrame:
    if len(rows) == 0:
        raise HTTPException(status_code=422, detail="rows must contain at least one item")

    df = pd.DataFrame(rows)
    feature_cols = list(contract["inference_input_columns"])
    id_cols = list(contract["inference_id_columns"])
    allowed_cols = set(feature_cols) | set(id_cols)

    training_only = sorted(set(df.columns) & set(TRAINING_ONLY_COLUMNS))
    if training_only:
        raise HTTPException(
            status_code=422,
            detail=f"Training-only columns are not allowed: {training_only}",
        )

    extra_cols = sorted(set(df.columns) - allowed_cols)
    if extra_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected request columns: {extra_cols}",
        )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required inference columns: {missing}",
        )

    for col in id_cols:
        if col not in df.columns:
            df[col] = None

    return df[id_cols + feature_cols]


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


def _load_json_file(path: Path, not_found_detail: str) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=not_found_detail)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_saved_scored_predictions(repo_root: Path) -> pd.DataFrame:
    runtime_root = resolve_runtime_root(repo_root)
    pred_path = runtime_root / "outputs" / "predictions" / "predictions_all.parquet"
    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Saved scored predictions not found. Run the offline scoring pipeline before calling /simulate-policy.",
        )
    df = pd.read_parquet(pred_path)
    required = {"churn_90d", "policy_ml", "policy_net_benefit", "value_pos"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Saved scored predictions are missing required simulation columns: {missing}",
        )
    return df


def _filter_saved_prediction_row(
    df: pd.DataFrame, customer_id: str, invoice_month: str
) -> pd.DataFrame:
    use = df.copy()
    if "invoice_month" in use.columns:
        use["invoice_month"] = use["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    matched = use[
        (use["CustomerID"].astype(str) == str(customer_id))
        & (use["invoice_month"] == str(invoice_month))
    ].copy()
    if len(matched) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No saved scored row found for CustomerID={customer_id}, invoice_month={invoice_month}.",
        )
    return matched.head(1)


def _load_model_summary(repo_root: Path) -> dict[str, Any]:
    runtime_root = resolve_runtime_root(repo_root)
    manifest_path = runtime_root / "reports" / "training_manifest.json"
    comparison_path = runtime_root / "reports" / "model_comparison.csv"
    promotion_path = runtime_root / "models" / "promoted" / "production.json"

    manifest = _load_json_file(manifest_path, "training_manifest.json not found.")
    comparison = (
        pd.read_csv(comparison_path)
        if comparison_path.exists()
        else pd.DataFrame()
    )
    promotion = _load_json_file(promotion_path, "production.json not found.")

    best_model = manifest.get("best_model")
    comparison_row = None
    if best_model is not None and len(comparison):
        matched = comparison[comparison["model"] == best_model]
        if len(matched):
            comparison_row = matched.iloc[0].to_dict()

    return {
        "manifest": manifest,
        "promotion": promotion,
        "comparison_row": comparison_row,
    }


def _load_policy_metrics(repo_root: Path, model_name: str, split: str) -> pd.DataFrame:
    runtime_root = resolve_runtime_root(repo_root)
    path = runtime_root / "reports" / "evaluation" / f"{model_name}_{split}_policy_results.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Policy metrics not found for model={model_name}, split={split}.",
        )
    return pd.read_csv(path)


def _load_model_comparison(repo_root: Path) -> pd.DataFrame:
    runtime_root = resolve_runtime_root(repo_root)
    path = runtime_root / "reports" / "model_comparison.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_comparison.csv not found.")
    return pd.read_csv(path)


def _load_feature_importance(repo_root: Path) -> pd.DataFrame:
    runtime_root = resolve_runtime_root(repo_root)
    path = runtime_root / "reports" / "feature_importance.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="feature_importance.csv not found.")
    return pd.read_csv(path)


def _load_target_records(repo_root: Path, budget_pct: int) -> pd.DataFrame:
    runtime_root = resolve_runtime_root(repo_root)
    path = runtime_root / "outputs" / "targets" / f"targets_all_k{budget_pct:02d}.parquet"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Target list not found for budget {budget_pct}%.",
        )
    return pd.read_parquet(path)


def _load_latest_drift(repo_root: Path) -> dict[str, Any]:
    runtime_root = resolve_runtime_root(repo_root)
    drift_path = runtime_root / "reports" / "monitoring" / "drift_latest.json"
    return _load_json_file(drift_path, "drift_latest.json not found.")


def _load_drift_history_records(repo_root: Path, limit: int) -> dict[str, Any]:
    runtime_root = resolve_runtime_root(repo_root)
    history_path = runtime_root / "reports" / "monitoring" / "drift_history.csv"
    df = load_drift_history(history_path)
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="drift_history.csv not found.")
    ordered = df.sort_values("generated_at_utc", ascending=False)
    return {
        "total_rows": int(len(ordered)),
        "returned_rows": int(min(len(ordered), limit)),
        "rows": _serialize_records(ordered.head(limit)),
    }


def create_app(repo_root: Path | None = None) -> FastAPI:
    repo_root = repo_root or _repo_root_from_app_file()
    cfg = _load_api_config(repo_root)
    budgets = [float(x) for x in cfg.get("eval", {}).get("budgets", [])]
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns_store")
    decision_cfg = get_decision_policy_config(cfg)
    experiment_cfg = get_experiment_config(cfg)
    monitoring_cfg = get_monitoring_alert_config(cfg)
    llm_cfg = dict(cfg.get("llm", {}))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.repo_root = repo_root
        app.state.budgets = budgets
        app.state.decision_cfg = decision_cfg
        app.state.experiment_cfg = experiment_cfg
        app.state.monitoring_cfg = monitoring_cfg
        app.state.model_info = load_model(repo_root, tracking_uri)
        app.state.llm_agent = ChurnLLMAgent(
            tool_executor=ChurnToolExecutor(
                ToolContext(
                    repo_root=repo_root,
                    model_info=app.state.model_info,
                    budgets=budgets,
                    decision_cfg=decision_cfg,
                    experiment_cfg=experiment_cfg,
                    monitoring_cfg=monitoring_cfg,
                )
            ),
            llm_config=llm_cfg,
        )
        yield

    app = FastAPI(title="ChurnXGB API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_llm_router())

    @app.get("/health")
    def health() -> dict[str, Any]:
        model_info = app.state.model_info
        return {
            "status": "ok",
            "model_name": model_info["model_name"],
            "model_source": model_info["model_source"],
            "decision_simulation_assumption_driven": True,
        }

    @app.get("/model-summary")
    def model_summary() -> dict[str, Any]:
        return _load_model_summary(app.state.repo_root)

    @app.get("/policy-metrics")
    def policy_metrics(
        model_name: str | None = None,
        split: str = "test",
    ) -> dict[str, Any]:
        summary = _load_model_summary(app.state.repo_root)
        manifest = summary["manifest"]
        use_model = model_name or manifest.get("best_model")
        if use_model is None:
            raise HTTPException(status_code=404, detail="No best model found in manifest.")
        df = _load_policy_metrics(app.state.repo_root, str(use_model), split)
        return {
            "model_name": use_model,
            "split": split,
            "rows": _serialize_records(df),
        }

    @app.get("/model-comparison")
    def model_comparison() -> dict[str, Any]:
        df = _load_model_comparison(app.state.repo_root)
        return {"rows": _serialize_records(df)}

    @app.get("/feature-importance")
    def feature_importance(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        df = _load_feature_importance(app.state.repo_root)
        ordered = df.sort_values("importance", ascending=False)
        return {
            "total_rows": int(len(ordered)),
            "returned_rows": int(min(len(ordered), limit)),
            "rows": _serialize_records(ordered.head(limit)),
        }

    @app.get("/targets/{budget_pct}")
    def targets(
        budget_pct: int,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> dict[str, Any]:
        df = _load_target_records(app.state.repo_root, budget_pct)
        return {
            "budget_pct": int(budget_pct),
            "total_rows": int(len(df)),
            "returned_rows": int(min(len(df), limit)),
            "rows": _serialize_records(df.head(limit)),
        }

    @app.get("/drift/latest")
    def drift_latest() -> dict[str, Any]:
        return _load_latest_drift(app.state.repo_root)

    @app.get("/drift/history")
    def drift_history(
        limit: int = Query(default=50, ge=1, le=1000),
    ) -> dict[str, Any]:
        out = _load_drift_history_records(app.state.repo_root, limit)
        return {
            "monitoring_config": dict(app.state.monitoring_cfg),
            **out,
        }

    @app.get("/predictions")
    def predictions(
        limit: int = Query(default=100, ge=1, le=1000),
        sort_by: str = Query(default="policy_net_benefit"),
    ) -> dict[str, Any]:
        df = _load_saved_scored_predictions(app.state.repo_root)
        if sort_by not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"sort_by column not found: {sort_by}",
            )
        ordered = df.sort_values(sort_by, ascending=False)
        return {
            "sort_by": sort_by,
            "total_rows": int(len(ordered)),
            "returned_rows": int(min(len(ordered), limit)),
            "rows": _serialize_records(ordered.head(limit)),
        }

    @app.get("/customers/explain")
    def explain_saved_customer(
        customer_id: str,
        invoice_month: str,
        top_n: int = Query(default=5, ge=1, le=20),
    ) -> dict[str, Any]:
        scored = _load_saved_scored_predictions(app.state.repo_root)
        row = _filter_saved_prediction_row(scored, customer_id, invoice_month)
        model_info = app.state.model_info
        explanations = explain_prediction_rows(
            model=model_info["model"],
            X=row[model_info["feature_cols"]],
            feature_cols=model_info["feature_cols"],
            top_n=top_n,
        )
        prediction = _serialize_prediction_output(row)[0]
        return {
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

    @app.post("/predict")
    def predict(request: PredictRequest) -> list[dict[str, Any]]:
        model_info = app.state.model_info
        request_df = _prepare_request_frame(request.rows, model_info["contract"])
        scored = score_dataframe(
            request_df,
            model=model_info["model"],
            feature_cols=model_info["feature_cols"],
            contract=model_info["contract"],
            budgets=app.state.budgets,
            model_source=model_info["model_source"],
            decision_cfg=app.state.decision_cfg,
        )
        return _serialize_prediction_output(scored)

    @app.post("/explain")
    def explain(request: ExplainRequest) -> list[dict[str, Any]]:
        model_info = app.state.model_info
        request_df = _prepare_request_frame(request.rows, model_info["contract"])
        scored = score_dataframe(
            request_df,
            model=model_info["model"],
            feature_cols=model_info["feature_cols"],
            contract=model_info["contract"],
            budgets=app.state.budgets,
            model_source=model_info["model_source"],
            decision_cfg=app.state.decision_cfg,
        )
        explanations = explain_prediction_rows(
            model=model_info["model"],
            X=request_df[model_info["feature_cols"]],
            feature_cols=model_info["feature_cols"],
            top_n=request.top_n,
        )

        rows: list[dict[str, Any]] = []
        prediction_records = _serialize_prediction_output(scored)
        for idx, explanation in enumerate(explanations):
            row_out = {
                "identifiers": {
                    key: prediction_records[idx].get(key) for key in IDENTIFIER_COLUMNS
                },
                "prediction": {
                    key: prediction_records[idx].get(key)
                    for key in PREDICTION_OUTPUT_COLUMNS
                    if key not in IDENTIFIER_COLUMNS
                },
                **explanation,
            }
            rows.append(row_out)

        return rows

    @app.post("/simulate-policy")
    def simulate_policy(request: SimulatePolicyRequest) -> dict[str, Any]:
        use_budgets = (
            [float(x) for x in request.budgets]
            if request.budgets is not None
            else list(app.state.budgets)
        )
        scored = _load_saved_scored_predictions(app.state.repo_root)
        return {
            "assumption_driven": True,
            "note": "This endpoint simulates decision economics from flat configured intervention assumptions on saved offline scored predictions. It is assumption-driven threshold analysis, not causal inference or observed treatment effect estimation.",
            "decision_config": dict(app.state.decision_cfg),
            "results": simulate_policy_by_budget(scored, use_budgets),
        }

    @app.post("/simulate-experiment")
    def simulate_experiment(request: SimulateExperimentRequest) -> dict[str, Any]:
        use_budgets = (
            [float(x) for x in request.budgets]
            if request.budgets is not None
            else list(app.state.budgets)
        )
        scored = _load_saved_scored_predictions(app.state.repo_root)
        return {
            "assumption_driven": True,
            "note": "This endpoint runs a deterministic business-case simulation from configured treatment assumptions on saved offline scored predictions. It is not causal inference, not observed uplift estimation, and does not produce experimental confidence intervals.",
            "experiment_config": dict(app.state.experiment_cfg),
            "results": simulate_experiment_by_budget(
                scored,
                use_budgets,
                app.state.experiment_cfg,
            ),
        }

    return app


app = create_app()
