from __future__ import annotations

from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from churnxgb.baselines.heuristics import add_heuristics
from churnxgb.evaluation.backtest import run_backtest
from churnxgb.evaluation.classification import (
    classification_summary,
    curve_data,
    save_curve_data,
)
from churnxgb.evaluation.plots import (
    plot_calibration_curve,
    plot_lift_curve,
    plot_pr_curve,
    plot_roc_curve,
)
from churnxgb.evaluation.report import evaluate_policies
from churnxgb.modeling.interpretability import save_feature_importance_artifacts
from churnxgb.modeling.model_utils import save_model_artifacts
from churnxgb.modeling.promote import write_promotion_record
from churnxgb.modeling.train_models import train_and_predict
from churnxgb.monitoring.drift import build_reference_profile_with_counts
from churnxgb.split.temporal import temporal_split
from churnxgb.utils.hashing import sha256_file


def _resolve_tracking_uri(repo_root: Path, tracking_uri: str) -> str:
    if tracking_uri.startswith("file:./") or tracking_uri == "file:mlruns":
        abs_path = (repo_root / "mlruns").resolve()
        return "file:" + str(abs_path).replace("\\", "/")
    return tracking_uri


def _log_uplift(rep: pd.DataFrame, split: str, budgets: list[float]) -> None:
    for k in budgets:
        ml_row = rep[(rep["policy"] == "policy_ml") & (rep["budget_k"] == k)]
        if len(ml_row) != 1:
            continue
        ml_var = float(ml_row["value_at_risk"].iloc[0])

        base_rows = rep[
            (rep["policy"].isin(["policy_recency", "policy_rfm"]))
            & (rep["budget_k"] == k)
        ]
        if len(base_rows) == 0:
            continue
        best_base = float(base_rows["value_at_risk"].max())

        uplift_abs = ml_var - best_base
        uplift_pct = (uplift_abs / best_base) if best_base > 0 else 0.0

        step = int(float(k) * 100)
        mlflow.log_metric(f"{split}_uplift_abs", uplift_abs, step=step)
        mlflow.log_metric(f"{split}_uplift_pct", uplift_pct, step=step)


def _model_specs(cfg: dict) -> dict[str, dict]:
    xgb_params = cfg.get("model", {})
    return {
        "xgboost": xgb_params,
        "logistic_regression": {"max_iter": 1000, "C": 1.0, "random_state": 42},
        "lightgbm": {
            "n_estimators": xgb_params.get("n_estimators", 300),
            "learning_rate": xgb_params.get("learning_rate", 0.05),
            "subsample": xgb_params.get("subsample", 0.9),
            "colsample_bytree": xgb_params.get("colsample_bytree", 0.9),
            "random_state": xgb_params.get("random_state", 42),
            "num_leaves": 31,
        },
    }


def _feature_cols(train_df: pd.DataFrame) -> list[str]:
    drop_cols = {
        "CustomerID",
        "invoice_month",
        "T",
        "has_future_purchase_90d",
        "churn_90d",
        "customer_value_90d",
        "value_proxy",
        "value_pos",
        "recency_risk",
        "rfm_risk",
        "policy_ml",
        "policy_recency",
        "policy_rfm",
        "churn_prob",
        "r_rank",
        "f_rank",
        "m_rank",
        "prev_event_ts",
    }
    return [
        c
        for c in train_df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]


def _policy_metric_row(rep: pd.DataFrame, budget: float) -> pd.Series:
    row = rep[(rep["policy"] == "policy_ml") & (rep["budget_k"] == budget)]
    if len(row) != 1:
        raise ValueError(f"Expected a single ML policy row for budget={budget}.")
    return row.iloc[0]


def _write_markdown_table(df: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def _write_model_eval_summary(
    out_path: Path,
    best_model: str,
    chosen_budget: float,
    test_classification: pd.DataFrame,
    test_policy: pd.DataFrame,
) -> None:
    cls = test_classification[test_classification["model"] == best_model].iloc[0]
    pol = test_policy[(test_policy["model"] == best_model) & (test_policy["policy"] == "policy_ml")]
    pol = pol.sort_values("budget_k")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Model Evaluation Summary\n\n")
        f.write(
            f"Best promoted model: `{best_model}` selected by validation `value_at_risk` at {int(chosen_budget * 100)}% budget.\n\n"
        )
        f.write("## Test Classification Metrics\n\n")
        f.write(
            f"- ROC-AUC: {cls['roc_auc']:.4f}\n"
            f"- PR-AUC: {cls['pr_auc']:.4f}\n"
            f"- Brier score: {cls['brier_score']:.4f}\n"
            f"- Positive rate: {cls['positive_rate']:.4f}\n\n"
        )
        f.write("## Test Targeting Metrics (policy_ml)\n\n")
        f.write(pol[[
            "budget_k",
            "value_at_risk",
            "var_covered_frac",
            "precision_at_k",
            "recall_at_k",
            "lift_at_k",
            "captured_churners",
        ]].to_markdown(index=False))
        f.write("\n\n")
        f.write("## Figures\n\n")
        f.write("- `reports/figures/test_roc_curve.png`\n")
        f.write("- `reports/figures/test_pr_curve.png`\n")
        f.write("- `reports/figures/test_lift_curve.png`\n")
        f.write("- `reports/figures/test_calibration_curve.png`\n")


def _write_feature_analysis(
    out_path: Path,
    importance_label: str,
    fi_df: pd.DataFrame,
    best_model: str,
) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Feature Analysis\n\n")
        f.write(f"Primary model: `{best_model}`\n\n")
        f.write(f"Importance method: {importance_label}\n\n")
        f.write("Top features reflect the behavioral signals driving churn prioritization.\n\n")
        f.write(fi_df.head(15).to_markdown(index=False))
        f.write("\n")


def _write_backtest_summary(out_path: Path, backtest_summary: pd.DataFrame) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Backtest Summary\n\n")
        f.write(
            "Rolling expanding-window backtests reuse the canonical point-in-time feature table and retrain each model on multiple chronological windows.\n\n"
        )
        f.write(backtest_summary.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feats_path = repo_root / "data" / "processed" / "customer_month_features.parquet"
    df = pd.read_parquet(feats_path)
    data_version = sha256_file(feats_path)

    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = _resolve_tracking_uri(
        repo_root, mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "churnxgb"))

    budgets = [float(x) for x in cfg["eval"]["budgets"]]
    chosen_budget = 0.10 if 0.10 in budgets else budgets[0]

    reports_dir = repo_root / "reports"
    eval_dir = reports_dir / "evaluation"
    figures_dir = reports_dir / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = temporal_split(
        df,
        train_end=cfg["split"]["train_end"],
        val_start=cfg["split"]["val_start"],
        val_end=cfg["split"]["val_end"],
        test_start=cfg["split"]["test_start"],
        test_end=cfg["split"]["test_end"],
    )

    train_df = add_heuristics(train_df)
    val_df = add_heuristics(val_df)
    test_df = add_heuristics(test_df)

    feature_cols = _feature_cols(train_df)
    model_specs = _model_specs(cfg)

    comparison_rows: list[dict] = []
    classification_rows: list[dict] = []
    policy_rows: list[dict] = []
    trained_artifacts: dict[str, dict] = {}

    for model_name, params in model_specs.items():
        run_name = f"{mlflow_cfg.get('run_name_prefix', 'churn_model')}_{model_name}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("label_horizon_days", cfg["label"]["horizon_days"])
            mlflow.log_param("split_train_end", cfg["split"]["train_end"])
            mlflow.log_param("split_val_start", cfg["split"]["val_start"])
            mlflow.log_param("split_val_end", cfg["split"]["val_end"])
            mlflow.log_param("split_test_start", cfg["split"]["test_start"])
            mlflow.log_param("split_test_end", cfg["split"]["test_end"])
            mlflow.log_param("n_features", len(feature_cols))
            for key, value in params.items():
                mlflow.log_param(f"model_{key}", value)

            train_scored, val_scored, test_scored, model = train_and_predict(
                model_name=model_name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_cols=feature_cols,
                model_params=params,
            )
            train_scored = add_heuristics(train_scored)
            val_scored = add_heuristics(val_scored)
            test_scored = add_heuristics(test_scored)

            from churnxgb.policy.scoring import add_policy_scores

            train_scored = add_policy_scores(train_scored)
            val_scored = add_policy_scores(val_scored)
            test_scored = add_policy_scores(test_scored)

            model_artifact_name = f"churn_{model_name}_v1"
            meta = save_model_artifacts(
                repo_root, model, feature_cols, model_name=model_artifact_name
            )
            mlflow.sklearn.log_model(model, name="model")
            mlflow.log_artifact(meta["feature_cols_path"], artifact_path="schema")
            mlflow.log_artifact(meta["inference_contract_path"], artifact_path="schema")

            val_policy = evaluate_policies(val_scored, budgets)
            test_policy = evaluate_policies(test_scored, budgets)
            val_policy.insert(0, "model", model_name)
            test_policy.insert(0, "model", model_name)
            policy_rows.extend(val_policy.assign(split="val").to_dict("records"))
            policy_rows.extend(test_policy.assign(split="test").to_dict("records"))

            val_policy_path = eval_dir / f"{model_name}_val_policy_results.csv"
            test_policy_path = eval_dir / f"{model_name}_test_policy_results.csv"
            val_policy.to_csv(val_policy_path, index=False)
            test_policy.to_csv(test_policy_path, index=False)
            mlflow.log_artifact(str(val_policy_path), artifact_path="reports")
            mlflow.log_artifact(str(test_policy_path), artifact_path="reports")

            for split_name, scored_df in [("val", val_scored), ("test", test_scored)]:
                cls = classification_summary(scored_df["churn_90d"], scored_df["churn_prob"])
                classification_rows.append({"model": model_name, "split": split_name, **cls})
                for metric_name, metric_value in cls.items():
                    mlflow.log_metric(f"{split_name}_{metric_name}", float(metric_value))

                curves = curve_data(scored_df["churn_90d"], scored_df["churn_prob"])
                curve_paths = save_curve_data(curves, eval_dir, split_name, model_name)
                for curve_path in curve_paths.values():
                    mlflow.log_artifact(str(curve_path), artifact_path="reports")

            for split_name, rep in [("val", val_policy), ("test", test_policy)]:
                for k in budgets:
                    row = _policy_metric_row(rep, k)
                    step = int(k * 100)
                    for metric_name in [
                        "value_at_risk",
                        "var_covered_frac",
                        "precision_at_k",
                        "recall_at_k",
                        "lift_at_k",
                    ]:
                        mlflow.log_metric(
                            f"{split_name}_{metric_name}",
                            float(row[metric_name]),
                            step=step,
                        )
                _log_uplift(rep, split_name, budgets)

            val_row = _policy_metric_row(val_policy, chosen_budget)
            test_row = _policy_metric_row(test_policy, chosen_budget)
            val_cls = next(
                row for row in classification_rows if row["model"] == model_name and row["split"] == "val"
            )
            test_cls = next(
                row for row in classification_rows if row["model"] == model_name and row["split"] == "test"
            )

            comparison_rows.append(
                {
                    "model": model_name,
                    "run_id": run.info.run_id,
                    "chosen_budget": chosen_budget,
                    "val_value_at_risk": float(val_row["value_at_risk"]),
                    "test_value_at_risk": float(test_row["value_at_risk"]),
                    "val_var_covered_frac": float(val_row["var_covered_frac"]),
                    "test_var_covered_frac": float(test_row["var_covered_frac"]),
                    "val_precision_at_k": float(val_row["precision_at_k"]),
                    "test_precision_at_k": float(test_row["precision_at_k"]),
                    "val_recall_at_k": float(val_row["recall_at_k"]),
                    "test_recall_at_k": float(test_row["recall_at_k"]),
                    "val_lift_at_k": float(val_row["lift_at_k"]),
                    "test_lift_at_k": float(test_row["lift_at_k"]),
                    "val_roc_auc": float(val_cls["roc_auc"]),
                    "test_roc_auc": float(test_cls["roc_auc"]),
                    "val_pr_auc": float(val_cls["pr_auc"]),
                    "test_pr_auc": float(test_cls["pr_auc"]),
                    "val_brier_score": float(val_cls["brier_score"]),
                    "test_brier_score": float(test_cls["brier_score"]),
                    "model_registry_name": model_artifact_name,
                }
            )

            trained_artifacts[model_name] = {
                "run_id": run.info.run_id,
                "model": model,
                "model_registry_name": model_artifact_name,
                "train_scored": train_scored,
                "val_scored": val_scored,
                "test_scored": test_scored,
            }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["val_value_at_risk", "test_roc_auc"], ascending=[False, False]
    )
    classification_df = pd.DataFrame(classification_rows)
    policy_df = pd.DataFrame(policy_rows)

    comparison_path = reports_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    _write_markdown_table(comparison_df, reports_dir / "model_comparison.md", "Model Comparison")

    classification_path = eval_dir / "classification_metrics.csv"
    classification_df.to_csv(classification_path, index=False)
    policy_path = eval_dir / "policy_metrics_all_models.csv"
    policy_df.to_csv(policy_path, index=False)

    best_model = str(comparison_df.iloc[0]["model"])
    best_meta = trained_artifacts[best_model]
    best_run_id = str(best_meta["run_id"])
    best_registry_name = str(best_meta["model_registry_name"])

    best_test_curves = curve_data(
        best_meta["test_scored"]["churn_90d"], best_meta["test_scored"]["churn_prob"]
    )
    best_test_policy = policy_df[(policy_df["model"] == best_model) & (policy_df["split"] == "test")].copy()
    plot_roc_curve(best_test_curves["roc"], figures_dir / "test_roc_curve.png", f"{best_model} ROC")
    plot_pr_curve(best_test_curves["pr"], figures_dir / "test_pr_curve.png", f"{best_model} PR")
    plot_calibration_curve(
        best_test_curves["calibration"],
        figures_dir / "test_calibration_curve.png",
        f"{best_model} Calibration",
    )
    plot_lift_curve(best_test_policy[best_test_policy["policy"] == "policy_ml"], figures_dir / "test_lift_curve.png", f"{best_model} Lift@K")

    ref_path = reports_dir / "monitoring" / "reference_profile.json"
    build_reference_profile_with_counts(
        best_meta["train_scored"],
        feature_cols,
        ref_path,
        n_bins=10,
        include_score_col=None,
    )

    fi_df, importance_label, importance_plot_path = save_feature_importance_artifacts(
        best_meta["model"],
        best_meta["train_scored"][feature_cols],
        feature_cols,
        figures_dir,
        reports_dir / "feature_importance.csv",
    )
    _write_feature_analysis(
        reports_dir / "feature_analysis.md", importance_label, fi_df, best_model
    )

    prom_path = write_promotion_record(
        repo_root,
        best_run_id,
        model_name=best_registry_name,
        selection_metric=f"val_value_at_risk_at_{int(chosen_budget * 100)}",
        selection_value=float(comparison_df.iloc[0]["val_value_at_risk"]),
    )

    _write_model_eval_summary(
        reports_dir / "model_eval_summary.md",
        best_model,
        chosen_budget,
        classification_df[classification_df["split"] == "test"],
        policy_df[policy_df["split"] == "test"],
    )

    backtest_detail, backtest_summary = run_backtest(
        df, feature_cols, budgets, model_specs
    )
    backtest_detail.to_csv(reports_dir / "backtest_detail.csv", index=False)
    backtest_summary.to_csv(reports_dir / "backtest_summary.csv", index=False)
    _write_backtest_summary(reports_dir / "backtest_summary.md", backtest_summary)

    with open(reports_dir / "training_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model,
                "best_run_id": best_run_id,
                "best_registry_name": best_registry_name,
                "chosen_budget": chosen_budget,
                "data_version": data_version,
                "importance_plot": str(importance_plot_path),
                "promotion_record": str(prom_path),
            },
            f,
            indent=2,
        )

    print("\n=== TRAINING COMPLETE ===")
    print("best_model:", best_model)
    print("best_run_id:", best_run_id)
    print("best_registry_name:", best_registry_name)
    print("comparison_path:", comparison_path)
    print("policy_path:", policy_path)
    print("classification_path:", classification_path)
    print("backtest_summary:", reports_dir / "backtest_summary.csv")
    print("promotion_record:", prom_path)


if __name__ == "__main__":
    main()
