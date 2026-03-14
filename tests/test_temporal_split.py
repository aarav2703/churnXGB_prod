from __future__ import annotations

import pandas as pd

from churnxgb.split.temporal import temporal_split


def test_temporal_split_respects_month_boundaries() -> None:
    df = pd.DataFrame(
        {
            "invoice_month": pd.period_range("2010-01", periods=6, freq="M"),
            "x": range(6),
        }
    )

    train_df, val_df, test_df = temporal_split(
        df,
        train_end="2010-03",
        val_start="2010-04",
        val_end="2010-04",
        test_start="2010-05",
        test_end="2010-06",
    )

    assert train_df["invoice_month"].tolist() == list(pd.period_range("2010-01", "2010-03", freq="M"))
    assert val_df["invoice_month"].tolist() == [pd.Period("2010-04", freq="M")]
    assert test_df["invoice_month"].tolist() == list(pd.period_range("2010-05", "2010-06", freq="M"))
