"""
Rolling historical features prior to T.

We compute event-level rolling aggregates, then merge them onto the
customer_month cutoff timestamp T.

Features:
- rev_sum_30d / 90d / 180d
- freq_30d / 90d (event counts)
- rev_std_90d (volatility of event_revenue)
- return_count_90d (count events with negative revenue)
- aov_90d (avg event revenue)
"""

from __future__ import annotations

import pandas as pd


def _rolling_group(ev: pd.DataFrame, window: str, col: str, agg: str) -> pd.Series:
    """
    Grouped time-based rolling aggregation over InvoiceDate.

    Important: `.rolling(..., on=...)` requires rolling on a DataFrame, not a Series.
    We therefore roll on ev[['InvoiceDate', col]] and then select the aggregated col.
    """
    rolled = ev.groupby("CustomerID")[["InvoiceDate", col]].rolling(
        window=window, on="InvoiceDate"
    )

    if agg == "sum":
        out = rolled.sum()[col]
    elif agg == "count":
        out = rolled.count()[col]
    elif agg == "std":
        out = rolled.std()[col]
    elif agg == "mean":
        out = rolled.mean()[col]
    else:
        raise ValueError(f"Unknown agg={agg}")

    # out index aligns with ev after dropping group level
    return out.reset_index(level=0, drop=True)


def build_rolling_features(event_df: pd.DataFrame) -> pd.DataFrame:
    ev = event_df.copy()
    ev = ev.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    # helper columns for rolling
    ev["is_return"] = (ev["event_revenue"] < 0).astype(int)

    # Revenue sums
    ev["rev_sum_30d"] = _rolling_group(ev, "30D", "event_revenue", "sum")
    ev["rev_sum_90d"] = _rolling_group(ev, "90D", "event_revenue", "sum")
    ev["rev_sum_180d"] = _rolling_group(ev, "180D", "event_revenue", "sum")

    # Frequency (events count)
    ev["freq_30d"] = _rolling_group(ev, "30D", "event_revenue", "count")
    ev["freq_90d"] = _rolling_group(ev, "90D", "event_revenue", "count")

    # Volatility
    ev["rev_std_90d"] = _rolling_group(ev, "90D", "event_revenue", "std")

    # Returns behavior
    ev["return_count_90d"] = _rolling_group(ev, "90D", "is_return", "sum")

    # Average order value proxy
    ev["aov_90d"] = _rolling_group(ev, "90D", "event_revenue", "mean")

    feat_events = ev[
        [
            "CustomerID",
            "InvoiceDate",
            "rev_sum_30d",
            "rev_sum_90d",
            "rev_sum_180d",
            "freq_30d",
            "freq_90d",
            "rev_std_90d",
            "return_count_90d",
            "aov_90d",
        ]
    ].copy()

    return feat_events
