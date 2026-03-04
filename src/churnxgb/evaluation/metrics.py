"""
Decision-focused metrics: Value at Risk @ K% under a targeting policy.
"""

from __future__ import annotations

import pandas as pd


def value_at_risk_at_k(df: pd.DataFrame, policy_col: str, k: float) -> float:
    """
    Compute Value at Risk @ K under a given policy.

    Steps:
    1) rank rows by policy_col descending
    2) select top K% rows
    3) sum value_pos among rows that actually churned (churn_90d == 1)

    Requires:
    - churn_90d
    - value_pos
    - policy_col
    """
    if not (0 < k <= 1):
        raise ValueError("k must be in (0, 1].")

    use = df.copy()
    use = use.sort_values(policy_col, ascending=False)

    n = len(use)
    top_n = max(1, int(round(n * k)))
    top = use.iloc[:top_n]

    var = float(top.loc[top["churn_90d"] == 1, "value_pos"].sum())
    return var


def total_value_at_risk(df: pd.DataFrame) -> float:
    """
    Total possible value at risk in the split (if you could intervene on everyone):
    sum of value_pos among churned customers.
    """
    return float(df.loc[df["churn_90d"] == 1, "value_pos"].sum())
