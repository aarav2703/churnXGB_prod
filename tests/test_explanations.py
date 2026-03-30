from __future__ import annotations

import math

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churnxgb.modeling.interpretability import explain_prediction_rows


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


def test_logistic_row_explanations_reconstruct_prediction_logit() -> None:
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
    model = _trained_logistic_pipeline()
    row = pd.DataFrame(
        {
            "rev_sum_30d": [50.0],
            "rev_sum_90d": [80.0],
            "rev_sum_180d": [120.0],
            "freq_30d": [2.0],
            "freq_90d": [4.0],
            "rev_std_90d": [5.0],
            "return_count_90d": [0.0],
            "aov_90d": [20.0],
            "gap_days_prev": [12.0],
        }
    )

    explanation = explain_prediction_rows(model, row, feature_cols, top_n=3)[0]
    total_logit = explanation["base_value_logit"] + sum(
        item["contribution_logit"] for item in explanation["feature_contributions"]
    )

    assert explanation["explanation_method"] == "logistic_pipeline_logit_contributions"
    assert math.isclose(total_logit, explanation["prediction_logit"], rel_tol=1e-9, abs_tol=1e-9)
    assert 0.0 <= explanation["prediction_probability"] <= 1.0
