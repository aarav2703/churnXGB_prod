"""
Build deduplicated customer event table.

Grain: (CustomerID, InvoiceDate)
This is used for both labeling and feature engineering to avoid double-counting.
"""

from __future__ import annotations

import pandas as pd


def build_customer_events(invoice_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert invoice_df into an event-level table at (CustomerID, InvoiceDate).

    This aggregates invoices that share the exact same timestamp for a customer,
    preventing row inflation in later merges.

    Expected invoice_df columns:
      - CustomerID
      - InvoiceDate
      - invoice_total_revenue
      - invoice_total_quantity
      - num_lines

    Returns:
      event_df with unique (CustomerID, InvoiceDate)
    """
    required = {
        "CustomerID",
        "InvoiceDate",
        "invoice_total_revenue",
        "invoice_total_quantity",
        "num_lines",
    }
    missing = required - set(invoice_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in invoice_df: {missing}")

    g = invoice_df.groupby(["CustomerID", "InvoiceDate"], as_index=False)
    event_df = g.agg(
        event_revenue=("invoice_total_revenue", "sum"),
        event_quantity=("invoice_total_quantity", "sum"),
        event_num_invoices=("num_lines", "sum"),
        event_invoice_count=("CustomerID", "size"),
    )

    # Uniqueness guarantee
    if event_df.duplicated(["CustomerID", "InvoiceDate"]).any():
        raise AssertionError("Duplicate (CustomerID, InvoiceDate) found in event_df.")

    event_df = event_df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(
        drop=True
    )
    return event_df
