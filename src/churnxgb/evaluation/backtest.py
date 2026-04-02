from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from churnxgb.baselines.heuristics import add_heuristics
from churnxgb.evaluation.classification import classification_summary
from churnxgb.evaluation.report import evaluate_policies
from churnxgb.modeling.train_models import train_and_predict
from churnxgb.policy.scoring import add_policy_scores


@dataclass(frozen=True)
class BacktestFold:
    fold_name: str
    train_end: str
    test_start: str
    test_end: str


def build_expanding_window_folds(
    months: list[pd.Period],
    min_train_months: int = 6,
    test_window_months: int = 2,
    step_months: int = 2,
) -> list[BacktestFold]:
    folds: list[BacktestFold] = []
    i = min_train_months
    while i + test_window_months <= len(months):
        train_end = months[i - 1]
        test_start = months[i]
        test_end = months[i + test_window_months - 1]
        folds.append(
            BacktestFold(
                fold_name=f"{test_start}_{test_end}",
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
            )
        )
        i += step_months
    return folds


def run_backtest(
    df: pd.DataFrame,
    feature_cols: list[str],
    budgets: list[float],
    model_specs: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(df["invoice_month"].astype("period[M]").unique())
    folds = build_expanding_window_folds(months)

    rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for fold in folds:
        train_df = df[df["invoice_month"] <= pd.Period(fold.train_end, freq="M")].copy()
        test_df = df[
            (df["invoice_month"] >= pd.Period(fold.test_start, freq="M"))
            & (df["invoice_month"] <= pd.Period(fold.test_end, freq="M"))
        ].copy()

        train_df = add_heuristics(train_df)
        test_df = add_heuristics(test_df)

        for model_name, params in model_specs.items():
            train_scored, _, test_scored, _ = train_and_predict(
                model_name=model_name,
                train_df=train_df,
                val_df=test_df,
                test_df=test_df,
                feature_cols=feature_cols,
                model_params=params,
            )
            del train_scored
            test_scored = add_policy_scores(test_scored)

            cls = classification_summary(test_scored["churn_90d"], test_scored["churn_prob"])
            pol = evaluate_policies(test_scored, budgets)
            ml_rows = pol[pol["policy"] == "policy_ml"].copy()

            for _, row in ml_rows.iterrows():
                rows.append(
                    {
                        "fold": fold.fold_name,
                        "model": model_name,
                        "train_end": fold.train_end,
                        "test_start": fold.test_start,
                        "test_end": fold.test_end,
                        "budget_k": float(row["budget_k"]),
                        "value_at_risk": float(row["value_at_risk"]),
                        "net_benefit_at_k": float(row["net_benefit_at_k"])
                        if row.get("net_benefit_at_k") is not None
                        else None,
                        "var_covered_frac": float(row["var_covered_frac"]),
                        "precision_at_k": float(row["precision_at_k"]),
                        "recall_at_k": float(row["recall_at_k"]),
                        "lift_at_k": float(row["lift_at_k"]),
                        "roc_auc": cls["roc_auc"],
                        "pr_auc": cls["pr_auc"],
                        "brier_score": cls["brier_score"],
                    }
                )

    detail_df = pd.DataFrame(rows)
    grouped = (
        detail_df.groupby(["model", "budget_k"], as_index=False)[
            [
                "value_at_risk",
                "net_benefit_at_k",
                "var_covered_frac",
                "precision_at_k",
                "recall_at_k",
                "lift_at_k",
                "roc_auc",
                "pr_auc",
                "brier_score",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join([str(x) for x in col if x]).rstrip("_") for col in grouped.columns.to_flat_index()
    ]
    summary_df = grouped.rename(columns={"model_": "model", "budget_k_": "budget_k"})
    return detail_df, summary_df
