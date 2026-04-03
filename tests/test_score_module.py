from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pandas as pd

from churnxgb.artifacts import ArtifactPaths
from churnxgb.inference.contracts import build_inference_contract, PREDICTION_OUTPUT_COLUMNS
from churnxgb.modeling.model_utils import save_model_artifacts
from churnxgb.monitoring.drift import build_reference_profile_with_counts
from churnxgb.pipeline.score import (
    build_outputs,
    load_model,
    score_dataframe,
    simulate_policy_by_budget,
)


class DummyModel:
    def predict_proba(self, X: pd.DataFrame):
        probs = X["rev_sum_90d"].astype(float) / 100.0
        probs = probs.clip(lower=0.0, upper=1.0)
        return pd.concat([1.0 - probs, probs], axis=1).to_numpy()


ROOT = Path(__file__).resolve().parents[1]


def _make_repo_root() -> Path:
    path = ROOT / "tests_artifacts" / f"score_module_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": [1, 2],
            "invoice_month": pd.PeriodIndex(["2010-01", "2010-01"], freq="M"),
            "T": pd.to_datetime(["2010-01-31", "2010-01-31"]),
            "rev_sum_30d": [50.0, 20.0],
            "rev_sum_90d": [80.0, 10.0],
            "rev_sum_180d": [120.0, 25.0],
            "freq_30d": [2.0, 1.0],
            "freq_90d": [4.0, 1.0],
            "rev_std_90d": [5.0, 1.0],
            "return_count_90d": [0.0, 0.0],
            "aov_90d": [20.0, 10.0],
            "gap_days_prev": [3.0, 10.0],
        }
    )


def test_load_model_prefers_local_registry_when_no_promotion_record() -> None:
    repo_root = _make_repo_root()
    feature_cols = [
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
    save_model_artifacts(
        repo_root, DummyModel(), feature_cols, model_name="churn_xgb_v1"
    )

    try:
        info = load_model(repo_root, tracking_uri="sqlite:///mlflow.db")

        assert info["model_source"] == "local_registry:churn_xgb_v1"
        assert info["run_id"] is None
        assert info["contract"]["inference_input_columns"] == feature_cols
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_score_dataframe_adds_scores_and_target_flags() -> None:
    df = _base_frame()
    feature_cols = [
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
    contract = build_inference_contract(feature_cols)

    scored = score_dataframe(
        df,
        model=DummyModel(),
        feature_cols=feature_cols,
        contract=contract,
        budgets=[0.5],
        model_source="local_registry:churn_xgb_v1",
    )

    assert "churn_prob" in scored.columns
    assert "policy_ml" in scored.columns
    assert "assumed_success_rate_customer" in scored.columns
    assert "intervention_cost_customer" in scored.columns
    assert "expected_retained_value" in scored.columns
    assert "expected_cost" in scored.columns
    assert "policy_net_benefit" in scored.columns
    assert "decision_simulation_assumption_driven" in scored.columns
    assert "target_k50" in scored.columns
    assert int(scored["target_k50"].sum()) == 1


def test_score_dataframe_preserves_ranking_under_flat_decision_assumptions() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "invoice_month": pd.PeriodIndex(["2010-01", "2010-01", "2010-01"], freq="M"),
            "T": pd.to_datetime(["2010-01-31", "2010-01-31", "2010-01-31"]),
            "rev_sum_30d": [10.0, 50.0, 100.0],
            "rev_sum_90d": [20.0, 60.0, 80.0],
            "rev_sum_180d": [30.0, 90.0, 140.0],
            "freq_30d": [1.0, 2.0, 4.0],
            "freq_90d": [1.0, 2.0, 5.0],
            "rev_std_90d": [1.0, 2.0, 3.0],
            "return_count_90d": [0.0, 0.0, 0.0],
            "aov_90d": [10.0, 20.0, 16.0],
            "gap_days_prev": [30.0, 8.0, 2.0],
        }
    )
    feature_cols = [
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
    contract = build_inference_contract(feature_cols)

    scored = score_dataframe(
        df,
        model=DummyModel(),
        feature_cols=feature_cols,
        contract=contract,
        budgets=[1 / 3],
        model_source="local_registry:churn_xgb_v1",
        decision_cfg={
            "intervention_cost": 15.0,
            "assumed_success_rate": 0.15,
            "retention_value_multiplier": 1.0,
        },
    )

    baseline_order = scored.sort_values("policy_ml", ascending=False)["CustomerID"].tolist()
    comparison_order = scored.sort_values("policy_net_benefit", ascending=False)["CustomerID"].tolist()

    assert baseline_order == comparison_order


def test_simulate_policy_by_budget_reports_no_ranking_change_under_flat_assumptions() -> None:
    scored = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3],
            "policy_ml": [20.0, 19.0, 18.0],
            "policy_net_benefit": [5.0, 4.0, 3.0],
            "value_pos": [100.0, 80.0, 60.0],
            "churn_90d": [1, 1, 0],
        }
    )

    result = simulate_policy_by_budget(scored, [0.5])

    assert len(result) == 1
    assert result[0]["assumption_driven"] is True
    assert "comparison_net_benefit_at_k" in result[0]
    assert result[0]["ranking_changed"] is False
    assert result[0]["n_customers_rank_changed"] == 0
    assert result[0]["selected_customers_differ"] is False


def test_build_outputs_writes_prediction_artifacts() -> None:
    repo_root = _make_repo_root()
    scored = _base_frame().copy()
    scored["churn_prob"] = [0.8, 0.1]
    scored["value_pos"] = [80.0, 10.0]
    scored["policy_ml"] = [64.0, 1.0]
    scored["assumed_success_rate_customer"] = [0.2, 0.1]
    scored["intervention_cost_customer"] = [20.0, 10.0]
    scored["expected_retained_value"] = [12.0, 0.5]
    scored["expected_cost"] = [15.0, 15.0]
    scored["policy_net_benefit"] = [-3.0, -14.5]
    scored["decision_simulation_assumption_driven"] = [True, True]

    try:
        out = build_outputs(
            repo_root,
            scored,
            feature_cols=["rev_sum_90d"],
            budgets=[0.5],
            split_name="all",
        )

        pred_df = pd.read_parquet(out["inference_pred_path"])
        assert list(pred_df.columns) == PREDICTION_OUTPUT_COLUMNS
        assert out["target_paths"][0.5].exists()
        assert out["drift_report"] is None
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)


def test_build_outputs_appends_drift_history_and_alerts_when_reference_exists() -> None:
    repo_root = _make_repo_root()
    monitoring_dir = ArtifactPaths.for_repo(repo_root).monitoring_dir
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    reference_df = pd.DataFrame(
        {
            "rev_sum_90d": [10.0, 20.0, 30.0, 40.0],
            "gap_days_prev": [1.0, 2.0, 3.0, 4.0],
            "churn_prob": [0.1, 0.2, 0.3, 0.4],
        }
    )
    build_reference_profile_with_counts(
        reference_df,
        feature_cols=["rev_sum_90d", "gap_days_prev"],
        out_path=monitoring_dir / "reference_profile.json",
    )
    scored = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "invoice_month": pd.PeriodIndex(["2010-01"] * 4, freq="M"),
            "T": pd.to_datetime(["2010-01-31"] * 4),
            "rev_sum_90d": [100.0, 120.0, 140.0, 160.0],
            "gap_days_prev": [10.0, 12.0, 14.0, 16.0],
            "churn_prob": [0.8, 0.7, 0.6, 0.5],
            "value_pos": [80.0, 70.0, 60.0, 50.0],
            "policy_ml": [64.0, 49.0, 36.0, 25.0],
            "assumed_success_rate_customer": [0.15, 0.15, 0.15, 0.15],
            "intervention_cost_customer": [15.0, 15.0, 15.0, 15.0],
            "expected_retained_value": [9.6, 8.4, 5.4, 3.75],
            "expected_cost": [15.0, 15.0, 15.0, 15.0],
            "policy_net_benefit": [-5.4, -6.6, -9.6, -11.25],
            "decision_simulation_assumption_driven": [True, True, True, True],
        }
    )

    try:
        out = build_outputs(
            repo_root,
            scored,
            feature_cols=["rev_sum_90d", "gap_days_prev"],
            budgets=[0.5],
            split_name="all",
        )

        assert out["drift_report"] is not None
        assert out["drift_alerts"] is not None
        assert out["drift_history_path"].exists()
        hist = pd.read_csv(out["drift_history_path"])
        assert len(hist) == 1
        assert "overall_status" in hist.columns
    finally:
        shutil.rmtree(repo_root, ignore_errors=True)
