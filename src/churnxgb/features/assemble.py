"""
Assemble core modeling grains (customer-month) and helper columns.
"""

from __future__ import annotations

import pandas as pd


def add_invoice_month(invoice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add invoice_month (Period[M]) column.
    """
    df = invoice_df.copy()
    df["invoice_month"] = df["InvoiceDate"].dt.to_period("M")
    return df


def build_customer_month(invoice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the customer-month table (one row per customer per month).

    - invoice_month is derived from InvoiceDate
    - T is defined as last purchase timestamp in that month (max InvoiceDate)

    Returns columns:
      - CustomerID
      - invoice_month (Period[M])
      - T (datetime)
    """
    df = add_invoice_month(invoice_df)

    g = df.groupby(["CustomerID", "invoice_month"], as_index=False)
    customer_month = g.agg(T=("InvoiceDate", "max"))

    # Uniqueness guarantee
    if customer_month.duplicated(["CustomerID", "invoice_month"]).any():
        raise AssertionError(
            "Duplicate (CustomerID, invoice_month) found in customer_month."
        )

    customer_month = customer_month.sort_values(
        ["CustomerID", "invoice_month"]
    ).reset_index(drop=True)
    return customer_month
