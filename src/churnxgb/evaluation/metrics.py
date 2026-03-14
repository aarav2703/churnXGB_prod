"""
Decision-focused and ranking metrics for budget-constrained targeting.
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


def top_k_classification_metrics(
    df: pd.DataFrame, ranking_col: str, k: float
) -> dict[str, float]:
    """
    Compute targeting-oriented classification metrics in the top-K slice.

    Metrics are computed after ranking by `ranking_col` descending:
    - targeted_count
    - captured_churners
    - precision_at_k
    - recall_at_k
    - lift_at_k
    """
    if not (0 < k <= 1):
        raise ValueError("k must be in (0, 1].")

    use = df.sort_values(ranking_col, ascending=False).copy()
    top_n = max(1, int(round(len(use) * k)))
    top = use.iloc[:top_n]

    positives_total = int(use["churn_90d"].sum())
    captured = int(top["churn_90d"].sum())
    precision = float(captured / top_n) if top_n > 0 else 0.0
    recall = float(captured / positives_total) if positives_total > 0 else 0.0
    base_rate = float(positives_total / len(use)) if len(use) > 0 else 0.0
    lift = float(precision / base_rate) if base_rate > 0 else 0.0

    return {
        "targeted_count": float(top_n),
        "captured_churners": float(captured),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
    }
