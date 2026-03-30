from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _history_row_from_report(
    report: dict[str, Any],
    alert_summary: dict[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    features = report.get("features", {})
    psi_rows = [
        {"feature": feature, "psi": info.get("psi")}
        for feature, info in features.items()
        if info.get("psi") is not None
    ]
    psi_rows.sort(key=lambda row: float(row["psi"]), reverse=True)
    top_feature = psi_rows[0]["feature"] if psi_rows else None
    max_psi = float(psi_rows[0]["psi"]) if psi_rows else None
    summary = report.get("summary", {})
    score_stats = report.get("score_current_stats", {})

    return {
        "generated_at_utc": generated_at_utc,
        "n_rows_current": int(report.get("n_rows_current", 0)),
        "n_ok": int(summary.get("n_ok", 0)),
        "n_warn": int(summary.get("n_warn", 0)),
        "n_alert": int(summary.get("n_alert", 0)),
        "n_missing_ref": int(summary.get("n_missing_ref", 0)),
        "overall_status": alert_summary.get("overall_status"),
        "has_alerts": bool(alert_summary.get("has_alerts", False)),
        "has_warnings": bool(alert_summary.get("has_warnings", False)),
        "n_alert_features": int(alert_summary.get("n_alert_features", 0)),
        "n_warn_features": int(alert_summary.get("n_warn_features", 0)),
        "top_alert_feature": top_feature,
        "top_psi": max_psi,
        "score_mean": score_stats.get("mean"),
        "score_p90": score_stats.get("p90"),
        "score_p99": score_stats.get("p99"),
    }


def append_drift_history(
    report: dict[str, Any],
    alert_summary: dict[str, Any],
    history_path: Path,
    generated_at_utc: str,
) -> Path:
    hist = build_drift_history_frame(
        history_path, report, alert_summary, generated_at_utc
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(history_path, index=False)
    return history_path


def load_drift_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


def build_drift_history_frame(
    history_path: Path,
    report: dict[str, Any],
    alert_summary: dict[str, Any],
    generated_at_utc: str,
) -> pd.DataFrame:
    row = _history_row_from_report(report, alert_summary, generated_at_utc)
    new_df = pd.DataFrame([row])

    if history_path.exists():
        hist = pd.read_csv(history_path)
        return pd.concat([hist, new_df], ignore_index=True)
    return new_df
