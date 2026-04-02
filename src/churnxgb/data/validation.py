from __future__ import annotations

from typing import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def validate_null_thresholds(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    df_name: str,
) -> None:
    for column, threshold in thresholds.items():
        if column not in df.columns:
            continue
        null_rate = float(df[column].isna().mean())
        if null_rate > float(threshold):
            raise ValueError(
                f"{df_name} column '{column}' null rate {null_rate:.3f} exceeded threshold {threshold:.3f}."
            )


def validate_unique_keys(df: pd.DataFrame, keys: list[str], df_name: str) -> None:
    require_columns(df, keys, df_name)
    if df.duplicated(keys).any():
        raise ValueError(f"{df_name} has duplicate keys on {keys}.")


def validate_date_order(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    df_name: str,
) -> None:
    require_columns(df, [group_col, date_col], df_name)
    ordered = df.sort_values([group_col, date_col]).copy()
    deltas = ordered.groupby(group_col)[date_col].diff()
    if deltas.dropna().lt(pd.Timedelta(0)).any():
        raise ValueError(f"{df_name} contains decreasing {date_col} values within {group_col}.")

