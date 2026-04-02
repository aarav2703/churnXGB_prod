from __future__ import annotations

import pandas as pd
import pytest

from churnxgb.data.validation import (
    require_columns,
    validate_date_order,
    validate_null_thresholds,
    validate_unique_keys,
)


def test_require_columns_raises_for_missing_columns() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        require_columns(df, ["a", "b"], "example_df")


def test_validate_null_thresholds_raises_when_exceeded() -> None:
    df = pd.DataFrame({"a": [1.0, None, None]})
    with pytest.raises(ValueError):
        validate_null_thresholds(df, {"a": 0.5}, "example_df")


def test_validate_unique_keys_raises_for_duplicates() -> None:
    df = pd.DataFrame({"id": [1, 1], "month": ["2010-01", "2010-01"]})
    with pytest.raises(ValueError):
        validate_unique_keys(df, ["id", "month"], "example_df")


def test_validate_date_order_accepts_sorted_groups() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2],
            "InvoiceDate": pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-01"]),
        }
    )
    validate_date_order(df, "CustomerID", "InvoiceDate", "example_df")
