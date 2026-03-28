from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pandas as pd

from churnxgb.inference.contracts import build_inference_contract, PREDICTION_OUTPUT_COLUMNS
from churnxgb.modeling.model_utils import save_model_artifacts
from churnxgb.pipeline.score import build_outputs, load_model, score_dataframe


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
    assert "target_k50" in scored.columns
    assert int(scored["target_k50"].sum()) == 1


def test_build_outputs_writes_prediction_artifacts() -> None:
    repo_root = _make_repo_root()
    scored = _base_frame().copy()
    scored["churn_prob"] = [0.8, 0.1]
    scored["value_pos"] = [80.0, 10.0]
    scored["policy_ml"] = [64.0, 1.0]

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
