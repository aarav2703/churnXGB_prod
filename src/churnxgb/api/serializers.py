from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import HTTPException

from churnxgb.inference.contracts import (
    IDENTIFIER_COLUMNS,
    PREDICTION_OUTPUT_COLUMNS,
    TRAINING_ONLY_COLUMNS,
    build_prediction_output,
)


def prepare_request_frame(rows: list[dict[str, Any]], contract: dict) -> pd.DataFrame:
    if len(rows) == 0:
        raise HTTPException(status_code=422, detail="rows must contain at least one item")

    df = pd.DataFrame(rows)
    feature_cols = list(contract["inference_input_columns"])
    id_cols = list(contract["inference_id_columns"])
    allowed_cols = set(feature_cols) | set(id_cols)

    training_only = sorted(set(df.columns) & set(TRAINING_ONLY_COLUMNS))
    if training_only:
        raise HTTPException(
            status_code=422,
            detail=f"Training-only columns are not allowed: {training_only}",
        )

    extra_cols = sorted(set(df.columns) - allowed_cols)
    if extra_cols:
        raise HTTPException(
            status_code=422,
            detail=f"Unexpected request columns: {extra_cols}",
        )

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required inference columns: {missing}",
        )

    for col in id_cols:
        if col not in df.columns:
            df[col] = None

    return df[id_cols + feature_cols]


def serialize_prediction_output(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = build_prediction_output(df).copy()

    if "invoice_month" in out.columns:
        out["invoice_month"] = out["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    if "T" in out.columns:
        out["T"] = out["T"].map(
            lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
        )

    out = out.where(pd.notna(out), None)
    return out[PREDICTION_OUTPUT_COLUMNS].to_dict(orient="records")


def serialize_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    if "invoice_month" in out.columns:
        out["invoice_month"] = out["invoice_month"].map(
            lambda x: None if pd.isna(x) else str(x)
        )
    if "T" in out.columns:
        out["T"] = out["T"].map(
            lambda x: None if pd.isna(x) else pd.Timestamp(x).isoformat()
        )
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def customer_prediction_payload(prediction_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "identifiers": {key: prediction_record.get(key) for key in IDENTIFIER_COLUMNS},
        "prediction": {
            key: prediction_record.get(key)
            for key in PREDICTION_OUTPUT_COLUMNS
            if key not in IDENTIFIER_COLUMNS
        },
    }
