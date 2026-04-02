from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd


def _psi_from_counts(
    expected: np.ndarray, actual: np.ndarray, eps: float = 1e-8
) -> float:
    expected = expected.astype(float) + eps
    actual = actual.astype(float) + eps
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _compute_bins(reference: pd.Series, n_bins: int = 10) -> list[float]:
    """
    Bin edges using quantiles on reference distribution.
    Returns edges including -inf/+inf.
    """
    s = reference.dropna().astype(float)
    if s.empty:
        return [-np.inf, np.inf]

    qs = np.linspace(0, 1, n_bins + 1)
    edges = list(np.unique(np.quantile(s, qs)))

    if len(edges) < 2:
        mn, mx = float(s.min()), float(s.max())
        if mn == mx:
            edges = [mn - 1.0, mx + 1.0]
        else:
            edges = [mn, mx]

    # Use interior edges, add +/- inf bounds
    edges = [-np.inf] + edges[1:-1] + [np.inf]
    return [float(x) for x in edges]


def build_reference_profile_with_counts(
    df_ref: pd.DataFrame,
    feature_cols: list[str],
    out_path: Path,
    n_bins: int = 10,
    include_score_col: str | None = None,
) -> Path:
    """
    Save reference distributions with histogram counts so PSI is well-defined.

    Stores PSI reference ONLY for feature_cols by default.
    Score drift for churn_prob is stored as summary stats (not PSI).
    """
    ref: dict[str, Any] = {"features": {}, "meta": {}}
    ref["meta"]["n_rows"] = int(len(df_ref))
    ref["meta"]["n_bins"] = int(n_bins)

    cols = list(feature_cols)
    if include_score_col and include_score_col in df_ref.columns:
        cols.append(include_score_col)

    for col in cols:
        s = df_ref[col]
        bins = _compute_bins(s, n_bins=n_bins)

        vals = s.dropna().astype(float).values
        counts, _ = np.histogram(vals, bins=bins)

        ref["features"][col] = {
            "missing_rate": float(s.isna().mean()),
            "bins": bins,
            "counts": [int(x) for x in counts],
        }

    # Score distribution reference stats (non-PSI)
    if "churn_prob" in df_ref.columns:
        s = df_ref["churn_prob"].dropna().astype(float)
        ref["meta"]["score_ref_stats"] = {
            "missing_rate": float(df_ref["churn_prob"].isna().mean()),
            "mean": float(s.mean()) if len(s) else None,
            "p50": float(np.quantile(s, 0.50)) if len(s) else None,
            "p90": float(np.quantile(s, 0.90)) if len(s) else None,
            "p99": float(np.quantile(s, 0.99)) if len(s) else None,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=2)

    return out_path


def drift_report(
    reference_path: Path,
    df_current: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold_warn: float = 0.1,
    psi_threshold_alert: float = 0.25,
    include_score_col: str | None = None,
) -> dict[str, Any]:
    """
    Compare current dataframe to reference using PSI + missingness.

    PSI is computed for feature_cols (and optionally for include_score_col if you pass it).
    churn_prob drift is always included as summary stats (mean/p50/p90/p99 + missingness).
    """
    with open(reference_path, "r", encoding="utf-8") as f:
        ref = json.load(f)

    out: dict[str, Any] = {
        "reference_path": str(reference_path),
        "n_rows_current": int(len(df_current)),
        "psi_threshold_warn": psi_threshold_warn,
        "psi_threshold_alert": psi_threshold_alert,
        "summary": {"n_ok": 0, "n_warn": 0, "n_alert": 0, "n_missing_ref": 0},
        "features": {},
    }

    cols = list(feature_cols)
    if include_score_col and include_score_col in df_current.columns:
        cols.append(include_score_col)

    for col in cols:
        if col not in df_current.columns:
            continue

        s_cur = df_current[col]
        miss_cur = float(s_cur.isna().mean())

        if col not in ref.get("features", {}):
            out["features"][col] = {
                "missing_rate_current": miss_cur,
                "psi": None,
                "status": "no_reference",
            }
            out["summary"]["n_missing_ref"] += 1
            continue

        bins = ref["features"][col]["bins"]
        expected_counts = np.array(ref["features"][col]["counts"], dtype=float)
        cur_counts, _ = np.histogram(s_cur.dropna().astype(float).values, bins=bins)

        psi_val = _psi_from_counts(expected_counts, cur_counts)

        status = "ok"
        if psi_val >= psi_threshold_alert:
            status = "alert"
            out["summary"]["n_alert"] += 1
        elif psi_val >= psi_threshold_warn:
            status = "warn"
            out["summary"]["n_warn"] += 1
        else:
            out["summary"]["n_ok"] += 1

        out["features"][col] = {
            "missing_rate_current": miss_cur,
            "psi": float(psi_val),
            "status": status,
        }

    # Score distribution stats (non-PSI)
    if "churn_prob" in df_current.columns:
        s = df_current["churn_prob"].dropna().astype(float)
        out["score_current_stats"] = {
            "missing_rate": float(df_current["churn_prob"].isna().mean()),
            "mean": float(s.mean()) if len(s) else None,
            "p50": float(np.quantile(s, 0.50)) if len(s) else None,
            "p90": float(np.quantile(s, 0.90)) if len(s) else None,
            "p99": float(np.quantile(s, 0.99)) if len(s) else None,
        }
        out["score_reference_stats"] = ref.get("meta", {}).get("score_ref_stats", None)

    return out


def top_psi_features(report: dict[str, Any], top_n: int = 10) -> list[dict[str, Any]]:
    items = []
    feats = report.get("features", {})
    for col, info in feats.items():
        psi_val = info.get("psi", None)
        if psi_val is None:
            continue
        items.append(
            {
                "feature": col,
                "psi": float(psi_val),
                "status": info.get("status", "unknown"),
            }
        )

    items.sort(key=lambda x: x["psi"], reverse=True)
    return items[:top_n]


def compute_decision_drift(
    scored_df: pd.DataFrame,
    budgets: list[float],
    ranking_col: str,
) -> pd.DataFrame:
    if "invoice_month" not in scored_df.columns:
        raise ValueError("Expected invoice_month to compute decision drift.")

    rows: list[dict[str, Any]] = []
    use = scored_df.copy()
    use["invoice_month"] = use["invoice_month"].astype("period[M]").astype(str)

    for month, month_df in use.groupby("invoice_month"):
        month_df = month_df.sort_values(ranking_col, ascending=False).reset_index(drop=True)
        total_rows = len(month_df)
        for k in budgets:
            top_n = max(1, int(round(total_rows * float(k))))
            top = month_df.head(top_n).copy()
            rows.append(
                {
                    "invoice_month": month,
                    "budget_k": float(k),
                    "ranking_policy": ranking_col,
                    "n_rows": int(total_rows),
                    "selected_count": int(top_n),
                    "selected_share": float(top_n / total_rows) if total_rows > 0 else 0.0,
                    "avg_churn_prob_top_k": float(top["churn_prob"].mean()) if "churn_prob" in top.columns else None,
                    "avg_value_pos_top_k": float(top["value_pos"].mean()) if "value_pos" in top.columns else None,
                    "var_at_k": float(top.loc[top["churn_90d"] == 1, "value_pos"].sum())
                    if {"churn_90d", "value_pos"}.issubset(top.columns)
                    else None,
                    "avg_policy_net_benefit_top_k": float(top["policy_net_benefit"].mean())
                    if "policy_net_benefit" in top.columns
                    else None,
                }
            )
    return pd.DataFrame(rows).sort_values(["invoice_month", "budget_k"]).reset_index(drop=True)
