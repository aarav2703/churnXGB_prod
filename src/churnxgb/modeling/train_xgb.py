"""
Train XGBoost churn probability model and score splits.
"""

from __future__ import annotations

import pandas as pd
from xgboost import XGBClassifier


def train_xgb_and_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "churn_90d",
    model_params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, XGBClassifier]:
    X_train = train_df[feature_cols]
    y_train = train_df[label_col]

    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    params = model_params or {}
    model = XGBClassifier(
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

    model.fit(X_train, y_train)

    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    train_out["churn_prob"] = model.predict_proba(X_train)[:, 1]
    val_out["churn_prob"] = model.predict_proba(X_val)[:, 1]
    test_out["churn_prob"] = model.predict_proba(X_test)[:, 1]

    return train_out, val_out, test_out, model
