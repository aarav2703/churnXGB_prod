from __future__ import annotations

from pathlib import Path
import json


def write_promotion_record(
    repo_root: Path,
    run_id: str,
    model_name: str = "churn_xgb_v1",
    selection_metric: str | None = None,
    selection_value: float | None = None,
) -> Path:
    """
    Write a lightweight 'production pointer' so scoring can use the promoted run.
    """
    out_dir = repo_root / "models" / "promoted"
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = {
        "run_id": run_id,
        "model_name": model_name,
        "registry_path": str(repo_root / "models" / "registry" / model_name),
    }
    if selection_metric is not None:
        rec["selection_metric"] = selection_metric
    if selection_value is not None:
        rec["selection_value"] = float(selection_value)

    out_path = out_dir / "production.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    return out_path
