from __future__ import annotations

import pandas as pd


def add_value_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define the value proxy used for Value-at-Risk targeting.

    IMPORTANT:
    - Do NOT use future 90d revenue here (it becomes 0 for churners by definition).
    - Use a pre-T value proxy (e.g., trailing revenue).

    We use:
      value_proxy = rev_sum_90d  (trailing 90-day revenue as-of T)
      value_pos = clip(value_proxy, lower=0)
    """
    out = df.copy()

    if "rev_sum_90d" not in out.columns:
        raise ValueError("Expected rev_sum_90d to compute value_pos (value proxy).")

    out["value_proxy"] = out["rev_sum_90d"]
    out["value_pos"] = out["value_proxy"].clip(lower=0.0)

    return out


def add_policy_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Requires columns:
    - churn_prob (for ML policy)
    - recency_risk (baseline)
    - rfm_risk (baseline)
    - rev_sum_90d (for value proxy via add_value_pos)
    """
    out = add_value_pos(df)

    if "churn_prob" in out.columns:
        out["policy_ml"] = out["churn_prob"] * out["value_pos"]

    if "recency_risk" in out.columns:
        out["policy_recency"] = out["recency_risk"] * out["value_pos"]

    if "rfm_risk" in out.columns:
        out["policy_rfm"] = out["rfm_risk"] * out["value_pos"]

    return out
