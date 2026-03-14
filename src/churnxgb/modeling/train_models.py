from __future__ import annotations

from typing import Any

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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

    model.fit(X_train, y_train)

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["churn_prob"] = model.predict_proba(X_train)[:, 1]
    val_out["churn_prob"] = model.predict_proba(X_val)[:, 1]
    test_out["churn_prob"] = model.predict_proba(X_test)[:, 1]

    return train_out, val_out, test_out, model
