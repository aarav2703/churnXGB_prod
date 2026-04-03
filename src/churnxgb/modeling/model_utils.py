from __future__ import annotations

from pathlib import Path
import json
import joblib

from churnxgb.artifacts import ArtifactPaths
from churnxgb.inference.contracts import write_inference_contract
from churnxgb.utils.io import atomic_joblib_dump, atomic_write_json


def save_model_artifacts(
    repo_root: Path,
    model,
    feature_cols: list[str],
    model_name: str = "churn_xgb_v1",
) -> dict:
    """
    Save model + feature columns under the canonical runtime registry.

    Returns metadata dict including paths.
    """
    artifacts = ArtifactPaths.for_repo(repo_root)
    out_dir = artifacts.model_registry_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    calibration_path = out_dir / "calibrator.joblib"
    feats_path = out_dir / "feature_cols.json"
    meta_path = out_dir / "metadata.json"
    contract_path = write_inference_contract(repo_root, model_name, feature_cols)

    atomic_joblib_dump(model, model_path)
    calibration_obj = getattr(model, "calibrator", None)
    if calibration_obj is not None:
        atomic_joblib_dump(calibration_obj, calibration_path)
    atomic_write_json(feats_path, feature_cols)

    meta = {
        "model_name": model_name,
        "model_path": str(model_path),
        "calibration_path": str(calibration_path) if calibration_obj is not None else None,
        "calibration_method": getattr(calibration_obj, "method", None),
        "calibration_metadata": getattr(model, "calibration_metadata", None),
        "feature_cols_path": str(feats_path),
        "inference_contract_path": str(contract_path),
    }
    atomic_write_json(meta_path, meta)

    return meta


def load_model_artifacts(
    repo_root: Path, model_name: str = "churn_xgb_v1"
) -> tuple[object, list[str], dict]:
    """
    Load model + feature columns from the canonical runtime registry.
    """
    base = ArtifactPaths.for_repo(repo_root).model_registry_dir(model_name)
    model = joblib.load(base / "model.joblib")

    with open(base / "feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(base / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, feature_cols, meta
