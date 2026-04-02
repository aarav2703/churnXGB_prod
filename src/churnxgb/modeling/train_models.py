from __future__ import annotations

from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churnxgb.modeling.calibration import CalibratedModel, ProbabilityCalibrator


def build_model(model_name: str, model_params: dict | None = None) -> Any:
    params = model_params or {}

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_lambda=params.get("reg_lambda", 1.0),
            random_state=params.get("random_state", 42),
            eval_metric="logloss",
            n_jobs=-1,
        )

    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=params.get("max_iter", 1000),
                        C=params.get("C", 1.0),
                        solver=params.get("solver", "lbfgs"),
                        random_state=params.get("random_state", 42),
                    ),
                ),
            ]
        )

    if model_name == "lightgbm":
        return LGBMClassifier(
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            num_leaves=params.get("num_leaves", 31),
            subsample=params.get("subsample", 0.9),
            colsample_bytree=params.get("colsample_bytree", 0.9),
            reg_lambda=params.get("reg_lambda", 0.0),
            random_state=params.get("random_state", 42),
            objective="binary",
            verbosity=-1,
        )

    raise ValueError(f"Unsupported model_name: {model_name}")


def train_and_predict(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "churn_90d",
    model_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    model = build_model(model_name, model_params=model_params)

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    calibration_method = (model_params or {}).get("calibration_method", "platt")

    train_use = train_df.copy()
    calibration_df = train_use.copy()
    if "invoice_month" in train_use.columns and train_use["invoice_month"].nunique() >= 3:
        months = sorted(train_use["invoice_month"].astype("period[M]").unique())
        calibration_months = months[max(1, int(len(months) * 0.8)) :]
        if len(calibration_months) >= 1:
            calibration_df = train_use[train_use["invoice_month"].isin(calibration_months)].copy()
            train_use = train_use[~train_use["invoice_month"].isin(calibration_months)].copy()
    if len(train_use) == 0 or len(calibration_df) == 0 or calibration_df[label_col].nunique() < 2:
        split_idx = max(1, int(len(train_df) * 0.8))
        train_use = train_df.iloc[:split_idx].copy()
        calibration_df = train_df.iloc[split_idx:].copy()
    if len(train_use) == 0 or len(calibration_df) == 0 or calibration_df[label_col].nunique() < 2:
        train_use = train_df.copy()
        calibration_df = train_df.copy()

    model.fit(train_use[feature_cols], train_use[label_col])

    raw_calibration_scores = model.predict_proba(calibration_df[feature_cols])[:, 1]
    calibrator = ProbabilityCalibrator(method=calibration_method).fit(
        raw_calibration_scores, calibration_df[label_col]
    )
    calibrated_model = CalibratedModel(
        base_model=model,
        calibrator=calibrator,
        calibration_metadata={
            "method": calibration_method,
            "calibration_rows": int(len(calibration_df)),
            "train_rows": int(len(train_use)),
        },
    )

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["churn_prob_raw"] = model.predict_proba(X_train)[:, 1]
    val_out["churn_prob_raw"] = model.predict_proba(X_val)[:, 1]
    test_out["churn_prob_raw"] = model.predict_proba(X_test)[:, 1]

    train_out["churn_prob"] = calibrated_model.predict_proba(X_train)[:, 1]
    val_out["churn_prob"] = calibrated_model.predict_proba(X_val)[:, 1]
    test_out["churn_prob"] = calibrated_model.predict_proba(X_test)[:, 1]

    return train_out, val_out, test_out, calibrated_model
