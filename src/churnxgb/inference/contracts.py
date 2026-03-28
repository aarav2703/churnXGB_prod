from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


IDENTIFIER_COLUMNS = ["CustomerID", "invoice_month", "T"]
TRAINING_ONLY_COLUMNS = [
    "has_future_purchase_90d",
    "churn_90d",
    "customer_value_90d",
]
PREDICTION_OUTPUT_COLUMNS = [
    "CustomerID",
    "invoice_month",
    "T",
    "churn_prob",
    "value_pos",
    "policy_ml",
]


def build_inference_contract(feature_cols: list[str]) -> dict:
    return {
        "inference_input_columns": list(feature_cols),
        "inference_id_columns": list(IDENTIFIER_COLUMNS),
        "training_only_columns": list(TRAINING_ONLY_COLUMNS),
        "prediction_output_columns": list(PREDICTION_OUTPUT_COLUMNS),
    }


def write_inference_contract(
    repo_root: Path, model_name: str, feature_cols: list[str]
) -> Path:
    out_path = repo_root / "models" / "registry" / model_name / "inference_contract.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    contract = build_inference_contract(feature_cols)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    return out_path


def load_inference_contract(
    repo_root: Path, model_name: str, feature_cols: list[str] | None = None
) -> dict:
    path = repo_root / "models" / "registry" / model_name / "inference_contract.json"
    if not path.exists():
        if feature_cols is None:
            raise FileNotFoundError(f"Inference contract not found: {path}")
        write_inference_contract(repo_root, model_name, feature_cols)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_inference_frame(df: pd.DataFrame, feature_cols: list[str]) -> None:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing inference feature columns: {missing}")

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Inference feature columns must be numeric: {non_numeric}")


def build_prediction_output(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in PREDICTION_OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing prediction output columns: {missing}")

    return df[PREDICTION_OUTPUT_COLUMNS].copy()
