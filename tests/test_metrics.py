from __future__ import annotations

import pandas as pd

from churnxgb.evaluation.classification import classification_summary
from churnxgb.evaluation.report import evaluate_policies


def test_policy_metrics_and_classification_metrics_sanity() -> None:
    df = pd.DataFrame(
        {
            "churn_90d": [1, 0, 1, 0],
            "value_pos": [100.0, 80.0, 50.0, 10.0],
            "policy_ml": [0.9, 0.8, 0.2, 0.1],
            "policy_recency": [0.8, 0.7, 0.3, 0.2],
            "policy_rfm": [0.85, 0.4, 0.3, 0.1],
            "churn_prob": [0.9, 0.7, 0.4, 0.2],
        }
    )

    rep = evaluate_policies(df, [0.5])
    ml_row = rep[rep["policy"] == "policy_ml"].iloc[0]

    assert ml_row["value_at_risk"] == 100.0
    assert ml_row["captured_churners"] == 1.0
    assert 0.0 <= ml_row["precision_at_k"] <= 1.0
    assert ml_row["lift_at_k"] >= 1.0

    cls = classification_summary(df["churn_90d"], df["churn_prob"])
    assert 0.0 <= cls["roc_auc"] <= 1.0
    assert 0.0 <= cls["pr_auc"] <= 1.0
    assert cls["brier_score"] >= 0.0
