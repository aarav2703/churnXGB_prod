from __future__ import annotations

from typing import Any


DEFAULT_ALERT_CONFIG = {
    "warn_threshold": 0.1,
    "alert_threshold": 0.25,
}


def get_monitoring_alert_config(cfg: dict | None = None) -> dict[str, float]:
    monitoring_cfg = (cfg or {}).get("monitoring", {})
    out = {
        "warn_threshold": float(
            monitoring_cfg.get(
                "warn_threshold",
                DEFAULT_ALERT_CONFIG["warn_threshold"],
            )
        ),
        "alert_threshold": float(
            monitoring_cfg.get(
                "alert_threshold",
                DEFAULT_ALERT_CONFIG["alert_threshold"],
            )
        ),
    }
    if out["warn_threshold"] < 0 or out["alert_threshold"] < 0:
        raise ValueError("monitoring thresholds must be >= 0.")
    if out["warn_threshold"] > out["alert_threshold"]:
        raise ValueError("warn_threshold must be <= alert_threshold.")
    return out


def summarize_drift_alerts(report: dict[str, Any]) -> dict[str, Any]:
    features = report.get("features", {})
    alert_features: list[dict[str, Any]] = []
    warn_features: list[dict[str, Any]] = []

    for feature, info in features.items():
        status = info.get("status")
        item = {
            "feature": feature,
            "psi": info.get("psi"),
            "status": status,
        }
        if status == "alert":
            alert_features.append(item)
        elif status == "warn":
            warn_features.append(item)

    overall_status = "ok"
    if alert_features:
        overall_status = "alert"
    elif warn_features:
        overall_status = "warn"

    return {
        "assumption_driven": False,
        "overall_status": overall_status,
        "has_alerts": bool(alert_features),
        "has_warnings": bool(warn_features),
        "n_alert_features": int(len(alert_features)),
        "n_warn_features": int(len(warn_features)),
        "alert_features": alert_features,
        "warn_features": warn_features,
    }
