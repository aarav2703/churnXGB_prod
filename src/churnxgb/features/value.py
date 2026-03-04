"""
Customer value calculation for the next 90 days AFTER T.

Adds:
- customer_value_90d  (sum of positive revenue in (T, T+90] )
"""

from __future__ import annotations

import pandas as pd


def add_customer_value_90d(
    feature_table: pd.DataFrame, event_df: pd.DataFrame, horizon_days: int = 90
) -> pd.DataFrame:
    ft = feature_table.copy()

    ev = event_df[["CustomerID", "InvoiceDate", "event_revenue"]].copy()
    ev = ev.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    # Prepare T and horizon endpoint
    ft["T_plus_h"] = ft["T"] + pd.to_timedelta(horizon_days, unit="D")

    merged = ft[["CustomerID", "invoice_month", "T", "T_plus_h"]].merge(
        ev,
        on="CustomerID",
        how="left",
    )

    in_window = (merged["InvoiceDate"] > merged["T"]) & (
        merged["InvoiceDate"] <= merged["T_plus_h"]
    )
    merged["revenue_in_window"] = merged["event_revenue"].where(in_window, 0.0)

    value_df = (
        merged.groupby(["CustomerID", "invoice_month"], as_index=False)[
            "revenue_in_window"
        ]
        .sum()
        .rename(columns={"revenue_in_window": "customer_value_90d"})
    )

    out = ft.merge(
        value_df, on=["CustomerID", "invoice_month"], how="left", validate="one_to_one"
    )
    out["customer_value_90d"] = out["customer_value_90d"].fillna(0.0)

    out = out.drop(columns=["T_plus_h"])
    return out
