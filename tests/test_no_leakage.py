from __future__ import annotations

import pandas as pd

from churnxgb.pipeline.train import _feature_cols


def test_training_feature_selection_excludes_future_label_columns() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1],
            "invoice_month": pd.PeriodIndex(["2010-01"], freq="M"),
            "T": pd.to_datetime(["2010-01-31"]),
            "churn_90d": [1],
            "has_future_purchase_90d": [0],
            "customer_value_90d": [25.0],
            "rev_sum_90d": [10.0],
            "freq_90d": [2.0],
            "gap_days_prev": [12.0],
        }
    )

    cols = _feature_cols(df)

    assert "customer_value_90d" not in cols
    assert "churn_90d" not in cols
    assert "has_future_purchase_90d" not in cols
    assert {"rev_sum_90d", "freq_90d", "gap_days_prev"}.issubset(cols)
