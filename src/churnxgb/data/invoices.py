"""
Invoice-level aggregation.

This builds invoice_df similar to the notebook: group raw transactions
into invoice-level records before building customer-month/event tables.
"""

from __future__ import annotations

import pandas as pd


def build_invoice_df(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build invoice-level dataframe.

    Aggregates transactions to invoice-level records with:
    - InvoiceDate (min timestamp per invoice)
    - CustomerID
    - invoice_total_revenue
    - invoice_total_quantity
    - num_lines

    Returns
    -------
    pd.DataFrame
        Invoice-level dataframe.
    """
    required = {"Invoice", "InvoiceDate", "CustomerID", "Quantity", "Revenue"}
    missing = required - set(df_clean.columns)
    if missing:
        raise ValueError(f"Missing required columns in cleaned df: {missing}")

    # Some datasets have Invoice as str-like; keep as-is.
    g = df_clean.groupby(["Invoice", "CustomerID"], as_index=False)

    invoice_df = g.agg(
        InvoiceDate=("InvoiceDate", "min"),
        invoice_total_revenue=("Revenue", "sum"),
        invoice_total_quantity=("Quantity", "sum"),
        num_lines=("Invoice", "size"),
    )

    invoice_df = invoice_df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(
        drop=True
    )
    return invoice_df
