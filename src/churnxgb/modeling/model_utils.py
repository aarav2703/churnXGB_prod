from __future__ import annotations

from pathlib import Path
import json
import joblib

from churnxgb.inference.contracts import write_inference_contract


def save_model_artifacts(
    repo_root: Path,
    model,
    feature_cols: list[str],
    model_name: str = "churn_xgb_v1",
) -> dict:
    """
    Save model + feature columns to models/registry/.

    Returns metadata dict including paths.
    """
    out_dir = repo_root / "models" / "registry" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    feats_path = out_dir / "feature_cols.json"
    meta_path = out_dir / "metadata.json"
    contract_path = write_inference_contract(repo_root, model_name, feature_cols)

    joblib.dump(model, model_path)

    with open(feats_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    meta = {
        "model_name": model_name,
        "model_path": str(model_path),
        "feature_cols_path": str(feats_path),
        "inference_contract_path": str(contract_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_model_artifacts(
    repo_root: Path, model_name: str = "churn_xgb_v1"
) -> tuple[object, list[str], dict]:
    """
    Load model + feature columns from models/registry/<model_name>/.
    """
    import json

    base = repo_root / "models" / "registry" / model_name
    model = joblib.load(base / "model.joblib")

    with open(base / "feature_cols.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    with open(base / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return model, feature_cols, meta
