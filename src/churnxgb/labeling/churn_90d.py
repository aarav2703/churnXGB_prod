"""
Churn labeling: churn_90d after cutoff timestamp T.

A row is churned if the customer makes no purchase in the next `horizon_days`
strictly after T.
"""

from __future__ import annotations

import pandas as pd


def label_churn_90d(
    customer_month: pd.DataFrame,
    event_df: pd.DataFrame,
    horizon_days: int = 90,
) -> pd.DataFrame:
    """
    Add churn_90d label to customer_month.

    Parameters
    ----------
    customer_month:
      columns: CustomerID, invoice_month, T
    event_df:
      columns: CustomerID, InvoiceDate (as event timestamp)
    horizon_days:
      churn horizon in days

    Returns
    -------
    customer_month_labeled: same rows + churn_90d (0/1)
    """
    cm = customer_month.copy()
    ev = event_df[["CustomerID", "InvoiceDate"]].copy()
    ev = ev.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    if cm.duplicated(["CustomerID", "invoice_month"]).any():
        raise AssertionError("customer_month has duplicate keys.")
    if ev.duplicated(["CustomerID", "InvoiceDate"]).any():
        raise AssertionError("event_df has duplicate (CustomerID, InvoiceDate).")

    # For each (customer, T), determine if there exists an event in (T, T + horizon_days]
    # Approach: merge cm with events on CustomerID and filter by time window, then collapse.
    cm["T_plus_h"] = cm["T"] + pd.to_timedelta(horizon_days, unit="D")

    merged = cm[["CustomerID", "invoice_month", "T", "T_plus_h"]].merge(
        ev,
        on="CustomerID",
        how="left",
    )
    # Future events strictly after T and within horizon
    in_window = (merged["InvoiceDate"] > merged["T"]) & (
        merged["InvoiceDate"] <= merged["T_plus_h"]
    )
    merged["has_future_purchase_90d"] = in_window

    future_any = (
        merged.groupby(["CustomerID", "invoice_month"], as_index=False)[
            "has_future_purchase_90d"
        ]
        .any()
        .rename(columns={"has_future_purchase_90d": "has_future_purchase_90d"})
    )

    out = cm.merge(
        future_any,
        on=["CustomerID", "invoice_month"],
        how="left",
        validate="one_to_one",
    )
    out["has_future_purchase_90d"] = out["has_future_purchase_90d"].fillna(False)

    # churn=1 if no future purchase
    out["churn_90d"] = (~out["has_future_purchase_90d"]).astype(int)

    out = (
        out.drop(columns=["T_plus_h"])
        .sort_values(["CustomerID", "invoice_month"])
        .reset_index(drop=True)
    )
    return out
