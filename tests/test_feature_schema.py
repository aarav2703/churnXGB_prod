from __future__ import annotations

from pathlib import Path

import pandas as pd

from churnxgb.artifacts import ArtifactPaths


def test_processed_feature_schema_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    feats_path = ArtifactPaths.for_repo(root).feature_table_path()
    assert feats_path.exists()

    df = pd.read_parquet(feats_path)
    required = {
        "CustomerID",
        "invoice_month",
        "T",
        "churn_90d",
        "rev_sum_90d",
        "freq_90d",
        "gap_days_prev",
        "customer_value_90d",
    }
    assert required.issubset(df.columns)
    assert int(df.duplicated(["CustomerID", "invoice_month"]).sum()) == 0
