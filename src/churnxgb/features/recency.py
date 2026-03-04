"""
Recency features: gap in days since previous purchase event.

Adds:
- prev_event_ts
- gap_days_prev (fill 999 for first event)
"""

from __future__ import annotations

import pandas as pd


def add_recency_features(
    feature_table: pd.DataFrame, event_df: pd.DataFrame
) -> pd.DataFrame:
    ft = feature_table.copy()

    ev = event_df[["CustomerID", "InvoiceDate"]].copy()
    ev = ev.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    # previous event timestamp per customer
    ev["prev_event_ts"] = ev.groupby("CustomerID")["InvoiceDate"].shift(1)
    ev["gap_days_prev"] = (
        ev["InvoiceDate"] - ev["prev_event_ts"]
    ).dt.total_seconds() / (3600 * 24)

    # merge current event timestamp == T (we want recency at cutoff T)
    out = ft.merge(
        ev.rename(columns={"InvoiceDate": "T"})[
            ["CustomerID", "T", "prev_event_ts", "gap_days_prev"]
        ],
        on=["CustomerID", "T"],
        how="left",
        validate="one_to_one",
    )

    # fill first-event gaps with sentinel
    out["gap_days_prev"] = out["gap_days_prev"].fillna(999.0)

    return out
