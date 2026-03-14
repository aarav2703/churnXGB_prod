from __future__ import annotations

import pandas as pd

from churnxgb.evaluation.metrics import (
    top_k_classification_metrics,
    total_value_at_risk,
    value_at_risk_at_k,
)


def evaluate_policies(df: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    """
    Evaluate available policy columns on a split.

    Expected policy columns (if present):
      - policy_ml
      - policy_recency
      - policy_rfm

    Also evaluate a random baseline by shuffling rows deterministically.
    """
    policies = []
    for col in ["policy_ml", "policy_recency", "policy_rfm"]:
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
                    "total_value_at_risk": total_var,
                    "var_covered_frac": (var_k / total_var) if total_var > 0 else 0.0,
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
                "total_value_at_risk": total_var,
                "var_covered_frac": (var_k / total_var) if total_var > 0 else 0.0,
                **cls,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["budget_k", "value_at_risk"], ascending=[True, False]
    ).reset_index(drop=True)
