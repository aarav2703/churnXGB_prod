from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import HTTPException

from churnxgb.artifacts import ArtifactPaths
from churnxgb.monitoring.history import load_drift_history

from churnxgb.api.serializers import serialize_records


def load_json_file(path: Path, not_found_detail: str) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=not_found_detail)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_saved_scored_predictions(repo_root: Path) -> pd.DataFrame:
    pred_path = ArtifactPaths.for_repo(repo_root).predictions_path("all")
    if not pred_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Saved scored predictions not found. Run the offline scoring pipeline before calling this endpoint.",
        )
    df = pd.read_parquet(pred_path)
    required = {"churn_90d", "policy_ml", "policy_net_benefit", "value_pos"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Saved scored predictions are missing required columns: {missing}",
        )
    return df


def filter_saved_prediction_row(
    df: pd.DataFrame, customer_id: str, invoice_month: str
) -> pd.DataFrame:
    use = df.copy()
    if "invoice_month" in use.columns:
        use["invoice_month"] = use["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    matched = use[
        (use["CustomerID"].astype(str) == str(customer_id))
        & (use["invoice_month"] == str(invoice_month))
    ].copy()
    if len(matched) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No saved scored row found for CustomerID={customer_id}, invoice_month={invoice_month}.",
        )
    return matched.head(1)


def load_model_summary(repo_root: Path) -> dict[str, Any]:
    artifacts = ArtifactPaths.for_repo(repo_root)
    manifest = load_json_file(
        artifacts.training_manifest_path(), "training_manifest.json not found."
    )
    comparison_path = artifacts.model_comparison_path()
    promotion = load_json_file(
        artifacts.promotion_record_path(), "production.json not found."
    )

    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    best_model = manifest.get("best_model")
    comparison_row = None
    if best_model is not None and len(comparison):
        matched = comparison[comparison["model"] == best_model]
        if len(matched):
            comparison_row = matched.iloc[0].to_dict()

    return {
        "manifest": manifest,
        "promotion": promotion,
        "comparison_row": comparison_row,
    }


def load_policy_metrics(repo_root: Path, model_name: str, split: str) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).evaluation_dir / f"{model_name}_{split}_policy_results.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Policy metrics not found for model={model_name}, split={split}.",
        )
    return pd.read_csv(path)


def load_model_comparison(repo_root: Path) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).model_comparison_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="model_comparison.csv not found.")
    return pd.read_csv(path)


def load_feature_importance(repo_root: Path) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).reports_dir / "feature_importance.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="feature_importance.csv not found.")
    return pd.read_csv(path)


def load_target_records(repo_root: Path, budget_pct: int) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).target_list_path("all", budget_pct)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Target list not found for budget {budget_pct}%.",
        )
    return pd.read_parquet(path)


def load_latest_drift(repo_root: Path) -> dict[str, Any]:
    drift_path = ArtifactPaths.for_repo(repo_root).monitoring_dir / "drift_latest.json"
    return load_json_file(drift_path, "drift_latest.json not found.")


def load_decision_drift(repo_root: Path) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).reports_dir / "decision_drift.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="decision_drift.csv not found.")
    return pd.read_csv(path)


def load_segment_evaluation(repo_root: Path) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).reports_dir / "evaluation_segments.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="evaluation_segments.csv not found.")
    return pd.read_csv(path)


def load_backtest_detail(repo_root: Path) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).reports_dir / "backtest_detail.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="backtest_detail.csv not found.")
    return pd.read_csv(path)


def load_budget_frontier(repo_root: Path, model_name: str) -> pd.DataFrame:
    path = ArtifactPaths.for_repo(repo_root).evaluation_dir / f"{model_name}_test_frontier.csv"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Budget frontier not found for model={model_name}.",
        )
    return pd.read_csv(path)


def load_drift_history_records(repo_root: Path, limit: int) -> dict[str, Any]:
    history_path = ArtifactPaths.for_repo(repo_root).monitoring_dir / "drift_history.csv"
    df = load_drift_history(history_path)
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="drift_history.csv not found.")
    ordered = df.sort_values("generated_at_utc", ascending=False)
    return {
        "total_rows": int(len(ordered)),
        "returned_rows": int(min(len(ordered), limit)),
        "rows": serialize_records(ordered.head(limit)),
    }
