from __future__ import annotations

import pandas as pd

from churnxgb.inference.contracts import (
    PREDICTION_OUTPUT_COLUMNS,
    TRAINING_ONLY_COLUMNS,
    build_inference_contract,
    build_prediction_output,
    validate_inference_frame,
)


def test_inference_contract_defines_clean_input_and_output_schema() -> None:
    feature_cols = ["rev_sum_90d", "freq_90d", "gap_days_prev"]

    contract = build_inference_contract(feature_cols)

    assert contract["inference_input_columns"] == feature_cols
    assert set(contract["training_only_columns"]) == set(TRAINING_ONLY_COLUMNS)
    assert "churn_90d" not in contract["prediction_output_columns"]
    assert contract["prediction_output_columns"] == PREDICTION_OUTPUT_COLUMNS


def test_build_prediction_output_excludes_training_only_columns() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1],
            "invoice_month": pd.PeriodIndex(["2010-01"], freq="M"),
            "T": pd.to_datetime(["2010-01-31"]),
            "churn_prob": [0.42],
            "value_pos": [100.0],
            "policy_ml": [42.0],
            "assumed_success_rate_customer": [0.2],
            "intervention_cost_customer": [18.0],
            "expected_retained_value": [6.3],
            "expected_cost": [18.0],
            "policy_net_benefit": [-8.7],
            "decision_simulation_assumption_driven": [True],
            "churn_90d": [1],
            "customer_value_90d": [25.0],
        }
    )

    out = build_prediction_output(df)

    assert list(out.columns) == PREDICTION_OUTPUT_COLUMNS
    assert "churn_90d" not in out.columns
    assert "customer_value_90d" not in out.columns


def test_validate_inference_frame_requires_numeric_feature_columns() -> None:
    df = pd.DataFrame(
        {
            "rev_sum_90d": [100.0],
            "freq_90d": [2.0],
            "gap_days_prev": [12.0],
        }
    )

    validate_inference_frame(df, ["rev_sum_90d", "freq_90d", "gap_days_prev"])
