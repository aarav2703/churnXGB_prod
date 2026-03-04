"""
Heuristic baselines.

Adds:
- recency_risk (higher gap = higher risk)
- rfm_risk (rank-based combined risk score)
"""

from __future__ import annotations

import pandas as pd


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def add_recency_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # higher gap_days_prev => higher risk
    out["recency_risk"] = _minmax(out["gap_days_prev"])
    return out


def add_rfm_baseline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Proxy RFM components (all computed as-of T using your existing features)
    # Recency: gap_days_prev (higher worse)
    # Frequency: freq_90d (lower worse)
    # Monetary: rev_sum_90d (lower worse)

    # Convert to ranks; risk should be higher for "worse" customers
    out["r_rank"] = out["gap_days_prev"].rank(
        method="average", ascending=True
    )  # small gap good
    out["f_rank"] = out["freq_90d"].rank(
        method="average", ascending=False
    )  # high freq good
    out["m_rank"] = out["rev_sum_90d"].rank(
        method="average", ascending=False
    )  # high revenue good

    # Normalize ranks -> [0,1] and invert to represent risk
    r = _minmax(out["r_rank"])
    f = 1.0 - _minmax(out["f_rank"])
    m = 1.0 - _minmax(out["m_rank"])

    out["rfm_risk"] = (r + f + m) / 3.0
    return out


def add_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    out = add_recency_baseline(df)
    out = add_rfm_baseline(out)
    return out
