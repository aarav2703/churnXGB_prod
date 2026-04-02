from __future__ import annotations

import pandas as pd

from churnxgb.evaluation.metrics import (
    net_benefit_at_k,
    top_k_classification_metrics,
    total_value_at_risk,
    value_at_risk_at_k,
)
from churnxgb.evaluation.classification import classification_summary


def policy_frontier(df: pd.DataFrame, policy_col: str, budgets: list[float]) -> pd.DataFrame:
    rows = []
    total_var = total_value_at_risk(df)
    for k in budgets:
        var_k = value_at_risk_at_k(df, policy_col, k)
        cls = top_k_classification_metrics(df, policy_col, k)
        rows.append(
            {
                "policy": policy_col,
                "budget_k": float(k),
                "value_at_risk": float(var_k),
                "var_covered_frac": (float(var_k) / total_var) if total_var > 0 else 0.0,
                "net_benefit_at_k": net_benefit_at_k(df, policy_col, k)
                if "policy_net_benefit" in df.columns
                else None,
                **cls,
            }
        )
    return pd.DataFrame(rows)


def add_segment_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["segment_value_band"] = pd.cut(
        out["rev_sum_90d"].astype(float),
        bins=[-float("inf"), 50.0, 200.0, float("inf")],
        labels=["low_value", "mid_value", "high_value"],
    ).astype(str)
    out["segment_recency_bucket"] = pd.cut(
        out["gap_days_prev"].astype(float),
        bins=[-float("inf"), 30.0, 90.0, float("inf")],
        labels=["recent", "warming", "stale"],
    ).astype(str)
    out["segment_frequency_bucket"] = pd.cut(
        out["freq_90d"].astype(float),
        bins=[-float("inf"), 2.0, 5.0, float("inf")],
        labels=["low_frequency", "mid_frequency", "high_frequency"],
    ).astype(str)
    return out


def evaluate_segments(
    df: pd.DataFrame,
    budgets: list[float],
    split_name: str,
    model_name: str,
    policy_col: str = "policy_net_benefit",
) -> pd.DataFrame:
    use = add_segment_columns(df)
    rows: list[dict] = []
    segment_cols = [
        "segment_value_band",
        "segment_recency_bucket",
        "segment_frequency_bucket",
    ]
    chosen_budget = 0.10 if 0.10 in budgets else float(budgets[0])
    for segment_col in segment_cols:
        for segment_value, segment_df in use.groupby(segment_col):
            if len(segment_df) < 5 or segment_df["churn_90d"].nunique() < 2:
                continue
            cls = classification_summary(segment_df["churn_90d"], segment_df["churn_prob"])
            frontier = policy_frontier(segment_df, policy_col, budgets)
            budget_row = frontier[frontier["budget_k"] == chosen_budget]
            if len(budget_row) != 1:
                continue
            row = budget_row.iloc[0].to_dict()
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "segment_type": segment_col,
                    "segment_value": segment_value,
                    "n_rows": int(len(segment_df)),
                    "positive_rate": float(segment_df["churn_90d"].mean()),
                    "roc_auc": float(cls["roc_auc"]),
                    "pr_auc": float(cls["pr_auc"]),
                    "brier_score": float(cls["brier_score"]),
                    **row,
                }
            )
    return pd.DataFrame(rows)


def evaluate_policies(df: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    """
    Evaluate available policy columns on a split.

    Expected policy columns (if present):
      - policy_ml
      - policy_net_benefit
      - policy_recency
      - policy_rfm

    Also evaluate a random baseline by shuffling rows deterministically.
    """
    policies = []
    for col in ["policy_ml", "policy_net_benefit", "policy_recency", "policy_rfm"]:
        if col in df.columns:
            policies.append(col)

    rows = []
    total_var = total_value_at_risk(df)

    # Evaluate real policies
    for pol in policies:
        for k in budgets:
            var_k = value_at_risk_at_k(df, pol, k)
            cls = top_k_classification_metrics(df, pol, k)
            rows.append(
                {
                    "policy": pol,
                    "budget_k": k,
                    "value_at_risk": var_k,
                    "net_benefit_at_k": net_benefit_at_k(df, pol, k)
                    if "policy_net_benefit" in df.columns
                    else None,
                    "total_value_at_risk": total_var,
                    "var_covered_frac": (var_k / total_var) if total_var > 0 else 0.0,
                    "assumption_driven": bool("policy_net_benefit" in df.columns),
                    **cls,
                }
            )

    # Random baseline: shuffle deterministically, then treat that order as ranking
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # Create an explicit random score column on the shuffled copy
    shuffled["policy_random"] = range(len(shuffled), 0, -1)

    for k in budgets:
        var_k = value_at_risk_at_k(shuffled, "policy_random", k)
        cls = top_k_classification_metrics(shuffled, "policy_random", k)
        rows.append(
            {
                "policy": "policy_random",
                "budget_k": k,
                "value_at_risk": var_k,
                "net_benefit_at_k": net_benefit_at_k(shuffled, "policy_random", k)
                if "policy_net_benefit" in shuffled.columns
                else None,
                "total_value_at_risk": total_var,
                "var_covered_frac": (var_k / total_var) if total_var > 0 else 0.0,
                "assumption_driven": bool("policy_net_benefit" in shuffled.columns),
                **cls,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["budget_k", "value_at_risk"], ascending=[True, False]
    ).reset_index(drop=True)
