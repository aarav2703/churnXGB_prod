from __future__ import annotations

from pathlib import Path

from churnxgb.artifacts import ArtifactPaths
from churnxgb.utils.io import atomic_write_json


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
    artifacts = ArtifactPaths.for_repo(repo_root)
    out_dir = artifacts.promoted_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = {
        "run_id": run_id,
        "model_name": model_name,
        "registry_path": str(artifacts.model_registry_dir(model_name)),
    }
    if selection_metric is not None:
        rec["selection_metric"] = selection_metric
    if selection_value is not None:
        rec["selection_value"] = float(selection_value)

    out_path = artifacts.promotion_record_path()
    atomic_write_json(out_path, rec)

    return out_path
