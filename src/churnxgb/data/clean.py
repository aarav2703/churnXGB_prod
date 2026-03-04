"""
Cleaning and basic normalization.

This should mirror your notebook's early cleaning steps:
- parse InvoiceDate
- drop missing Customer ID
- compute revenue
- basic column normalization
"""

from __future__ import annotations

import pandas as pd


def clean_transactions(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw transactions.

    Expected columns (Online Retail II):
    - Invoice
    - StockCode
    - Description
    - Quantity
    - InvoiceDate
    - Price
    - Customer ID
    - Country

    Returns a cleaned dataframe with:
    - InvoiceDate parsed as datetime
    - Customer ID renamed to CustomerID (int)
    - Revenue computed as Quantity * Price
    """
    df = df_raw.copy()

    # --- Standardize column naming (non-destructive) ---
    # Keep original columns if present; add convenience alias columns.
    if "Customer ID" in df.columns and "CustomerID" not in df.columns:
        df["CustomerID"] = df["Customer ID"]

    # Parse InvoiceDate
    if "InvoiceDate" not in df.columns:
        raise ValueError("Expected column 'InvoiceDate' not found in raw data.")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Drop rows without customer id (matches your described process)
    if "CustomerID" not in df.columns:
        raise ValueError(
            "Expected customer id column ('Customer ID' or 'CustomerID') not found."
        )
    df = df.dropna(subset=["CustomerID"]).copy()

    # Cast CustomerID to int (common in Online Retail II)
    # Use astype(int) after dropna to avoid errors.
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Basic numeric coercions
    # Quantity and Price are used to compute revenue; coerce to numeric
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    else:
        raise ValueError("Expected column 'Quantity' not found in raw data.")

    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    else:
        raise ValueError("Expected column 'Price' not found in raw data.")

    # Compute revenue (returns produce negative revenue naturally via negative Quantity)
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Drop rows with invalid InvoiceDate or Quantity/Price after coercion
    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price"]).copy()

    # Sort for deterministic downstream merges/rolling windows
    df = df.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    return df
