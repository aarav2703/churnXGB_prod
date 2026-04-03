from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from churnxgb.evaluation.experiment_simulation import get_experiment_config
from churnxgb.monitoring.alerts import get_monitoring_alert_config
from churnxgb.policy.scoring import get_decision_policy_config
from churnxgb.pipeline.score import load_model


def repo_root_from_app_file() -> Path:
    return Path(__file__).resolve().parents[3]


def load_api_config(repo_root: Path) -> dict[str, Any]:
    cfg_path = repo_root / "config" / "config.yaml"
    if not cfg_path.exists():
        return {
            "eval": {"budgets": []},
            "mlflow": {"tracking_uri": "file:./mlruns_store"},
            "decision": get_decision_policy_config({}),
            "experiment": get_experiment_config({}),
            "monitoring": get_monitoring_alert_config({}),
        }

    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_app_state(repo_root: Path) -> dict[str, Any]:
    cfg = load_api_config(repo_root)
    budgets = [float(x) for x in cfg.get("eval", {}).get("budgets", [])]
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns_store")
    decision_cfg = get_decision_policy_config(cfg)
    experiment_cfg = get_experiment_config(cfg)
    monitoring_cfg = get_monitoring_alert_config(cfg)
    llm_cfg = dict(cfg.get("llm", {}))
    model_info = load_model(repo_root, tracking_uri)
    return {
        "repo_root": repo_root,
        "budgets": budgets,
        "decision_cfg": decision_cfg,
        "experiment_cfg": experiment_cfg,
        "monitoring_cfg": monitoring_cfg,
        "model_info": model_info,
        "llm_config": llm_cfg,
    }
