from __future__ import annotations

import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    test_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    All dates are strings like '2010-06' (YYYY-MM).
    """
    use = df.copy()
    if "invoice_month" not in use.columns:
        raise ValueError("Expected invoice_month in dataframe.")

    # Ensure Period[M]
    if not isinstance(use["invoice_month"].dtype, pd.PeriodDtype):
        use["invoice_month"] = use["invoice_month"].astype("period[M]")

    train_end_p = pd.Period(train_end, freq="M")
    val_start_p = pd.Period(val_start, freq="M")
    val_end_p = pd.Period(val_end, freq="M")
    test_start_p = pd.Period(test_start, freq="M")
    test_end_p = pd.Period(test_end, freq="M")

    train_df = use[use["invoice_month"] <= train_end_p].copy()
    val_df = use[
        (use["invoice_month"] >= val_start_p) & (use["invoice_month"] <= val_end_p)
    ].copy()
    test_df = use[
        (use["invoice_month"] >= test_start_p) & (use["invoice_month"] <= test_end_p)
    ].copy()

    return train_df, val_df, test_df
