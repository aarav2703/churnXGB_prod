from __future__ import annotations

from pathlib import Path
import shutil
import uuid
import json

import pandas as pd
import yaml
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churnxgb.api.app import create_app
from churnxgb.modeling.model_utils import save_model_artifacts
from churnxgb.pipeline.score import build_outputs


class DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        probs = X["rev_sum_90d"].astype(float) / 100.0
        probs = probs.clip(lower=0.0, upper=1.0)
        return pd.concat([1.0 - probs, probs], axis=1).to_numpy()


ROOT = Path(__file__).resolve().parents[1]


def _make_repo_root() -> Path:
    path = ROOT / "tests_artifacts" / f"api_app_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_config(repo_root: Path) -> None:
    config_dir = repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "eval": {"budgets": [0.05, 0.10]},
                "mlflow": {"tracking_uri": "sqlite:///mlflow.db"},
                "decision": {
                    "intervention_cost": 15.0,
                    "assumed_success_rate": 0.15,
                    "retention_value_multiplier": 1.0,
                },
            },
            f,
            sort_keys=False,
        )


def _feature_cols() -> list[str]:
    return [
        "rev_sum_30d",
        "rev_sum_90d",
        "rev_sum_180d",
        "freq_30d",
        "freq_90d",
        "rev_std_90d",
        "return_count_90d",
        "aov_90d",
        "gap_days_prev",
    ]


def _write_summary_artifacts(repo_root: Path) -> None:
    reports_dir = repo_root / "reports"
    eval_dir = reports_dir / "evaluation"
    monitoring_dir = reports_dir / "monitoring"
    targets_dir = repo_root / "outputs" / "targets"
    promoted_dir = repo_root / "models" / "promoted"
    reports_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)
    promoted_dir.mkdir(parents=True, exist_ok=True)

    with open(reports_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": "logistic_regression",
                "best_run_id": "run-123",
                "best_registry_name": "churn_xgb_v1",
                "chosen_budget": 0.1,
            },
            f,
        )

    pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "run_id": "run-123",
                "chosen_budget": 0.1,
                "val_value_at_risk": 100.0,
                "test_value_at_risk": 120.0,
            }
        ]
    ).to_csv(reports_dir / "model_comparison.csv", index=False)

    pd.DataFrame(
        [
            {
                "policy": "policy_ml",
                "budget_k": 0.1,
                "value_at_risk": 120.0,
                "net_benefit_at_k": 10.0,
            }
        ]
    ).to_csv(eval_dir / "logistic_regression_test_policy_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": "policy_net_benefit",
                "budget_k": 0.05,
                "value_at_risk": 60.0,
                "net_benefit_at_k": 8.0,
                "precision_at_k": 0.4,
                "recall_at_k": 0.2,
                "lift_at_k": 1.2,
                "targeted_count": 10,
                "captured_churners": 4,
                "var_covered_frac": 0.3,
            },
            {
                "policy": "policy_net_benefit",
                "budget_k": 0.1,
                "value_at_risk": 120.0,
                "net_benefit_at_k": 10.0,
                "precision_at_k": 0.45,
                "recall_at_k": 0.3,
                "lift_at_k": 1.25,
                "targeted_count": 20,
                "captured_churners": 9,
                "var_covered_frac": 0.5,
            },
        ]
    ).to_csv(eval_dir / "logistic_regression_test_frontier.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "split": "test",
                "segment_type": "segment_value_band",
                "segment_value": "high_value",
                "n_rows": 10,
                "positive_rate": 0.4,
                "roc_auc": 0.7,
                "pr_auc": 0.6,
                "brier_score": 0.2,
                "policy": "policy_net_benefit",
                "budget_k": 0.1,
                "value_at_risk": 120.0,
                "var_covered_frac": 0.5,
                "net_benefit_at_k": 10.0,
                "targeted_count": 2,
                "captured_churners": 1,
                "precision_at_k": 0.5,
                "recall_at_k": 0.25,
                "lift_at_k": 1.2,
            }
        ]
    ).to_csv(reports_dir / "evaluation_segments.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": "2010-01_2010-02",
                "model": "logistic_regression",
                "budget_k": 0.1,
                "value_at_risk": 100.0,
                "net_benefit_at_k": 12.0,
                "var_covered_frac": 0.4,
                "precision_at_k": 0.5,
                "recall_at_k": 0.2,
                "lift_at_k": 1.1,
                "roc_auc": 0.7,
                "pr_auc": 0.6,
                "brier_score": 0.2,
            }
        ]
    ).to_csv(reports_dir / "backtest_detail.csv", index=False)
    pd.DataFrame(
        [
            {
                "invoice_month": "2010-01",
                "budget_k": 0.1,
                "ranking_policy": "policy_net_benefit",
                "n_rows": 100,
                "selected_count": 10,
                "selected_share": 0.1,
                "avg_churn_prob_top_k": 0.8,
                "avg_value_pos_top_k": 90.0,
                "var_at_k": 120.0,
                "avg_policy_net_benefit_top_k": 10.0,
            }
        ]
    ).to_csv(reports_dir / "decision_drift.csv", index=False)

    pd.DataFrame(
        [
            {
                "CustomerID": 1,
                "invoice_month": pd.Period("2010-01", freq="M"),
                "T": pd.Timestamp("2010-01-31"),
                "churn_prob": 0.8,
                "value_pos": 80.0,
                "policy_ml": 64.0,
            }
        ]
    ).to_parquet(targets_dir / "targets_all_k10.parquet", index=False)

    predictions_dir = repo_root / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "CustomerID": 1,
                "invoice_month": pd.Period("2010-01", freq="M"),
                "T": pd.Timestamp("2010-01-31"),
                "rev_sum_30d": 50.0,
                "rev_sum_90d": 80.0,
                "rev_sum_180d": 120.0,
                "freq_30d": 2.0,
                "freq_90d": 4.0,
                "rev_std_90d": 5.0,
                "return_count_90d": 0.0,
                "aov_90d": 20.0,
                "gap_days_prev": 3.0,
                "churn_prob": 0.8,
                "value_pos": 80.0,
                "policy_ml": 64.0,
                "assumed_success_rate_customer": 0.22,
                "intervention_cost_customer": 15.0,
                "expected_retained_value": 14.08,
                "expected_cost": 15.0,
                "policy_net_benefit": -0.92,
                "decision_simulation_assumption_driven": True,
                "churn_90d": 1,
            }
        ]
    ).to_parquet(predictions_dir / "predictions_all.parquet", index=False)

    with open(monitoring_dir / "drift_latest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": "2026-03-30T00:00:00+00:00",
                "summary": {"n_ok": 1, "n_warn": 0, "n_alert": 0},
                "alerts": {
                    "overall_status": "ok",
                    "n_warn_features": 0,
                    "n_alert_features": 0,
                },
            },
            f,
        )
    pd.DataFrame(
        [
            {
                "generated_at_utc": "2026-03-29T00:00:00+00:00",
                "overall_status": "warn",
                "n_warn": 1,
                "n_alert": 0,
                "top_alert_feature": "rev_sum_90d",
                "top_psi": 0.14,
                "score_mean": 0.42,
            },
            {
                "generated_at_utc": "2026-03-30T00:00:00+00:00",
                "overall_status": "ok",
                "n_warn": 0,
                "n_alert": 0,
                "top_alert_feature": "gap_days_prev",
                "top_psi": 0.03,
                "score_mean": 0.39,
            },
        ]
    ).to_csv(monitoring_dir / "drift_history.csv", index=False)

    with open(promoted_dir / "production.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": "run-123",
                "model_name": "churn_xgb_v1",
                "registry_path": str(repo_root / "models" / "registry" / "churn_xgb_v1"),
            },
            f,
        )


def _trained_logistic_pipeline() -> Pipeline:
    X = pd.DataFrame(
        {
            "rev_sum_30d": [10.0, 60.0, 90.0, 15.0],
            "rev_sum_90d": [20.0, 80.0, 120.0, 30.0],
            "rev_sum_180d": [30.0, 110.0, 180.0, 40.0],
            "freq_30d": [1.0, 3.0, 5.0, 1.0],
            "freq_90d": [1.0, 4.0, 6.0, 1.0],
            "rev_std_90d": [2.0, 4.0, 7.0, 2.0],
            "return_count_90d": [0.0, 0.0, 1.0, 0.0],
            "aov_90d": [10.0, 20.0, 35.0, 12.0],
            "gap_days_prev": [40.0, 15.0, 5.0, 35.0],
        }
    )
    y = [1, 0, 0, 1]
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X, y)
    return model


def test_api_health_and_predict_flow() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(repo_root, DummyModel(), _feature_cols(), model_name="churn_xgb_v1")

    try:
        with TestClient(create_app(repo_root)) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"

            payload = {
                "rows": [
                    {
                        "CustomerID": 123,
                        "invoice_month": "2010-01",
                        "T": "2010-01-31T00:00:00",
                        "rev_sum_30d": 50.0,
                        "rev_sum_90d": 80.0,
                        "rev_sum_180d": 120.0,
                        "freq_30d": 2.0,
                        "freq_90d": 4.0,
                        "rev_std_90d": 5.0,
                        "return_count_90d": 0.0,
                        "aov_90d": 20.0,
                        "gap_days_prev": 3.0,
                    }
                ]
            }
            response = client.post("/predict", json=payload)

            assert response.status_code == 200
            body = response.json()
            assert list(body[0].keys()) == [
                "CustomerID",
                "invoice_month",
                "T",
                "churn_prob",
                "value_pos",
                "policy_ml",
                "assumed_success_rate_customer",
                "intervention_cost_customer",
                "expected_retained_value",
                "expected_cost",
                "policy_net_benefit",
                "decision_simulation_assumption_driven",
            ]
            assert body[0]["CustomerID"] == 123
            assert body[0]["invoice_month"] == "2010-01"
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_artifact_endpoints_return_structured_json() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    _write_summary_artifacts(repo_root)
    save_model_artifacts(
        repo_root,
        _trained_logistic_pipeline(),
        _feature_cols(),
        model_name="churn_xgb_v1",
    )

    try:
        with TestClient(create_app(repo_root)) as client:
            model_summary = client.get("/model-summary")
            assert model_summary.status_code == 200
            assert model_summary.json()["manifest"]["best_model"] == "logistic_regression"

            policy_metrics = client.get("/policy-metrics")
            assert policy_metrics.status_code == 200
            assert policy_metrics.json()["model_name"] == "logistic_regression"
            assert len(policy_metrics.json()["rows"]) == 1

            comparison = client.get("/model-comparison")
            assert comparison.status_code == 200
            assert len(comparison.json()["rows"]) == 1

            feature_importance_path = repo_root / "reports" / "feature_importance.csv"
            pd.DataFrame(
                [
                    {"feature": "rev_sum_90d", "importance": 0.42, "source": "shap_mean_abs"},
                    {"feature": "gap_days_prev", "importance": 0.31, "source": "shap_mean_abs"},
                ]
            ).to_csv(feature_importance_path, index=False)

            feature_importance = client.get("/feature-importance?limit=1")
            assert feature_importance.status_code == 200
            assert feature_importance.json()["returned_rows"] == 1

            targets = client.get("/targets/10?limit=1")
            assert targets.status_code == 200
            assert targets.json()["budget_pct"] == 10
            assert targets.json()["returned_rows"] == 1

            drift = client.get("/drift/latest")
            assert drift.status_code == 200
            assert drift.json()["summary"]["n_ok"] == 1
            assert drift.json()["alerts"]["overall_status"] == "ok"

            drift_history = client.get("/drift/history?limit=1")
            assert drift_history.status_code == 200
            assert drift_history.json()["returned_rows"] == 1
            assert drift_history.json()["rows"][0]["overall_status"] == "ok"

            predictions = client.get("/predictions?limit=1&sort_by=policy_ml")
            assert predictions.status_code == 200
            assert predictions.json()["returned_rows"] == 1

            segments = client.get("/segments")
            assert segments.status_code == 200
            assert len(segments.json()["rows"]) == 1

            backtest = client.get("/backtest?model_name=logistic_regression&budget_pct=10")
            assert backtest.status_code == 200
            assert len(backtest.json()["rows"]) == 1

            frontier = client.get("/frontier?model_name=logistic_regression")
            assert frontier.status_code == 200
            assert len(frontier.json()["rows"]) == 2

            decision_drift = client.get("/drift/decision?budget_pct=10")
            assert decision_drift.status_code == 200
            assert len(decision_drift.json()["rows"]) == 1

            explain_saved = client.get("/customers/explain?customer_id=1&invoice_month=2010-01")
            assert explain_saved.status_code == 200
            assert explain_saved.json()["identifiers"]["CustomerID"] == 1

            llm_explainer = client.post(
                "/llm/explain_customer",
                json={"customer_id": "1", "invoice_month": "2010-01", "top_n": 3},
            )
            assert llm_explainer.status_code == 200
            assert "answer" in llm_explainer.json()
            assert llm_explainer.json()["tools_used"][0]["name"] == "get_customer_decision_context"
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_explain_returns_row_level_contributors_for_logistic_pipeline() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(
        repo_root,
        _trained_logistic_pipeline(),
        _feature_cols(),
        model_name="churn_xgb_v1",
    )

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post(
                "/explain",
                json={
                    "rows": [
                        {
                            "CustomerID": 321,
                            "invoice_month": "2010-01",
                            "T": "2010-01-31T00:00:00",
                            "rev_sum_30d": 50.0,
                            "rev_sum_90d": 80.0,
                            "rev_sum_180d": 120.0,
                            "freq_30d": 2.0,
                            "freq_90d": 4.0,
                            "rev_std_90d": 5.0,
                            "return_count_90d": 0.0,
                            "aov_90d": 20.0,
                            "gap_days_prev": 12.0,
                        }
                    ],
                    "top_n": 3,
                },
            )

            assert response.status_code == 200
            body = response.json()
            assert len(body) == 1
            assert body[0]["identifiers"]["CustomerID"] == 321
            assert body[0]["prediction"]["churn_prob"] >= 0.0
            assert body[0]["explanation_method"] == "logistic_pipeline_logit_contributions"
            assert len(body[0]["top_positive_contributors"]) <= 3
            assert "feature_contributions" in body[0]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_simulate_policy_returns_assumption_driven_budget_summary() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(repo_root, DummyModel(), _feature_cols(), model_name="churn_xgb_v1")
    scored = pd.DataFrame(
        {
            "CustomerID": [123, 456],
            "invoice_month": pd.PeriodIndex(["2010-01", "2010-01"], freq="M"),
            "T": pd.to_datetime(["2010-01-31", "2010-01-31"]),
            "churn_prob": [0.8, 0.2],
            "value_pos": [80.0, 20.0],
            "policy_ml": [64.0, 4.0],
            "assumed_success_rate_customer": [0.15, 0.15],
            "intervention_cost_customer": [15.0, 15.0],
            "expected_retained_value": [9.6, 0.6],
            "expected_cost": [15.0, 15.0],
            "policy_net_benefit": [-5.4, -14.4],
            "decision_simulation_assumption_driven": [True, True],
            "churn_90d": [1, 0],
        }
    )
    build_outputs(repo_root, scored, feature_cols=["value_pos"], budgets=[0.5], split_name="all")

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post(
                "/simulate-policy",
                json={
                    "budgets": [0.5],
                },
            )

            assert response.status_code == 200
            body = response.json()
            assert body["assumption_driven"] is True
            assert body["results"][0]["budget_k"] == 0.5
            assert "comparison_net_benefit_at_k" in body["results"][0]
            assert body["results"][0]["ranking_changed"] is False
            assert body["results"][0]["pct_customers_rank_changed"] == 0.0
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_simulate_experiment_returns_structured_results() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(repo_root, DummyModel(), _feature_cols(), model_name="churn_xgb_v1")
    scored = pd.DataFrame(
        {
            "CustomerID": [123, 456, 789, 999],
            "invoice_month": pd.PeriodIndex(["2010-01", "2010-01", "2010-01", "2010-01"], freq="M"),
            "T": pd.to_datetime(["2010-01-31", "2010-01-31", "2010-01-31", "2010-01-31"]),
            "churn_prob": [0.8, 0.6, 0.4, 0.2],
            "value_pos": [80.0, 40.0, 20.0, 10.0],
            "policy_ml": [64.0, 24.0, 8.0, 2.0],
            "assumed_success_rate_customer": [0.15, 0.15, 0.15, 0.15],
            "intervention_cost_customer": [15.0, 15.0, 15.0, 15.0],
            "expected_retained_value": [9.6, 3.6, 1.2, 0.3],
            "expected_cost": [15.0, 15.0, 15.0, 15.0],
            "policy_net_benefit": [-5.4, -11.4, -13.8, -14.7],
            "decision_simulation_assumption_driven": [True, True, True, True],
            "churn_90d": [1, 1, 0, 0],
        }
    )
    build_outputs(repo_root, scored, feature_cols=["value_pos"], budgets=[0.5], split_name="all")

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post("/simulate-experiment", json={"budgets": [0.5]})

            assert response.status_code == 200
            body = response.json()
            assert body["assumption_driven"] is True
            assert body["results"][0]["budget_k"] == 0.5
            assert "incremental_retained_value" in body["results"][0]
            assert "uplift_probability_ci" not in body["results"][0]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_llm_query_routes_to_tool_and_returns_grounded_answer() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    _write_summary_artifacts(repo_root)
    save_model_artifacts(
        repo_root,
        _trained_logistic_pipeline(),
        _feature_cols(),
        model_name="churn_xgb_v1",
    )

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post(
                "/llm/query",
                json={
                    "query": "Who should I target at 10% budget?",
                    "include_raw_data": True,
                },
            )

            assert response.status_code == 200
            body = response.json()
            assert "answer" in body
            assert body["tools_used"][0]["name"] == "get_targets"
            assert body["raw_data"][0]["ok"] is True
            assert body["raw_data"][0]["data"]["budget_pct"] == 10
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_llm_query_can_answer_project_overview_questions() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    _write_summary_artifacts(repo_root)
    save_model_artifacts(
        repo_root,
        _trained_logistic_pipeline(),
        _feature_cols(),
        model_name="churn_xgb_v1",
    )

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post(
                "/llm/query",
                json={
                    "query": "How does this project work and how do I run it?",
                    "include_raw_data": True,
                },
            )

            assert response.status_code == 200
            body = response.json()
            assert "answer" in body
            assert body["tools_used"][0]["name"] == "get_project_overview"
            assert body["raw_data"][0]["ok"] is True
            assert body["raw_data"][0]["data"]["entrypoints"]["train"] == "python -m churnxgb.pipeline.train"
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_api_rejects_training_only_columns() -> None:
    repo_root = _make_repo_root()
    _write_config(repo_root)
    save_model_artifacts(repo_root, DummyModel(), _feature_cols(), model_name="churn_xgb_v1")

    try:
        with TestClient(create_app(repo_root)) as client:
            response = client.post(
                "/predict",
                json={
                    "rows": [
                        {
                            "rev_sum_30d": 50.0,
                            "rev_sum_90d": 80.0,
                            "rev_sum_180d": 120.0,
                            "freq_30d": 2.0,
                            "freq_90d": 4.0,
                            "rev_std_90d": 5.0,
                            "return_count_90d": 0.0,
                            "aov_90d": 20.0,
                            "gap_days_prev": 3.0,
                            "churn_90d": 1,
                        }
                    ]
                },
            )

            assert response.status_code == 422
            assert "Training-only columns" in response.json()["detail"]
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)
