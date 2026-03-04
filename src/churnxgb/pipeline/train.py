from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

import mlflow
import mlflow.sklearn

from churnxgb.utils.hashing import sha256_file
from churnxgb.split.temporal import temporal_split
from churnxgb.baselines.heuristics import add_heuristics
from churnxgb.modeling.train_xgb import train_xgb_and_predict
from churnxgb.policy.scoring import add_policy_scores
from churnxgb.evaluation.report import evaluate_policies
from churnxgb.modeling.model_utils import save_model_artifacts
from churnxgb.modeling.promote import write_promotion_record
from churnxgb.monitoring.drift import build_reference_profile_with_counts


def _resolve_tracking_uri(repo_root: Path, tracking_uri: str) -> str:
    """
    Make MLflow file tracking URI stable on Windows.
    If user provides file:./mlruns, resolve to an absolute path.
    """
    if tracking_uri.startswith("file:./") or tracking_uri == "file:mlruns":
        abs_path = (repo_root / "mlruns").resolve()
        return "file:" + str(abs_path).replace("\\", "/")
    return tracking_uri


def _log_uplift(rep: pd.DataFrame, split: str, budgets: list[float]) -> None:
    """
    Log uplift of ML vs best baseline (recency or rfm) at each budget.
    """
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feats_path = repo_root / "data" / "processed" / "customer_month_features.parquet"
    df = pd.read_parquet(feats_path)

    # --- MLflow setup ---
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    tracking_uri = _resolve_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "churnxgb"))

    # Data version hash (reproducibility)
    data_version = sha256_file(feats_path)

    # Split
    train_df, val_df, test_df = temporal_split(
        df,
        train_end=cfg["split"]["train_end"],
        val_start=cfg["split"]["val_start"],
        val_end=cfg["split"]["val_end"],
        test_start=cfg["split"]["test_start"],
        test_end=cfg["split"]["test_end"],
    )

    # Add baselines
    train_df = add_heuristics(train_df)
    val_df = add_heuristics(val_df)
    test_df = add_heuristics(test_df)

    # Feature columns (numeric-only, exclude ids/labels/targets/policies)
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
    }
    feature_cols = [
        c
        for c in train_df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # Model params (optional)
    model_params = cfg.get("model", {})

    run_name_prefix = mlflow_cfg.get("run_name_prefix", "churn_xgb")
    budgets = [float(x) for x in cfg["eval"]["budgets"]]

    with mlflow.start_run(run_name=f"{run_name_prefix}_v1") as run:
        run_id = run.info.run_id

        # Log key params + versions
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("label_horizon_days", cfg["label"]["horizon_days"])
        mlflow.log_param("split_train_end", cfg["split"]["train_end"])
        mlflow.log_param("split_val_start", cfg["split"]["val_start"])
        mlflow.log_param("split_val_end", cfg["split"]["val_end"])
        mlflow.log_param("split_test_start", cfg["split"]["test_start"])
        mlflow.log_param("split_test_end", cfg["split"]["test_end"])
        mlflow.log_param("n_features", len(feature_cols))

        for k, v in model_params.items():
            mlflow.log_param(f"model_{k}", v)

        # Train model + predict
        train_scored, val_scored, test_scored, model = train_xgb_and_predict(
            train_df, val_df, test_df, feature_cols, model_params=model_params
        )

        # Save model artifacts (local registry) + log to MLflow
        meta = save_model_artifacts(
            repo_root, model, feature_cols, model_name="churn_xgb_v1"
        )
        print("\n=== SAVED MODEL ===")
        print(meta)

        # Log model + schema artifacts
        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_artifact(meta["feature_cols_path"], artifact_path="schema")

        # Add policies
        train_scored = add_policy_scores(train_scored)
        val_scored = add_policy_scores(val_scored)
        test_scored = add_policy_scores(test_scored)

        # Drift reference profile from TRAIN split (PSI only for features; churn_prob tracked via stats)
        ref_path = repo_root / "reports" / "monitoring" / "reference_profile.json"
        build_reference_profile_with_counts(
            train_scored,
            feature_cols,
            ref_path,
            n_bins=10,
            include_score_col=None,
        )
        mlflow.log_artifact(str(ref_path), artifact_path="monitoring")
        print("\n=== DRIFT REFERENCE WRITTEN ===")
        print(ref_path)

        # Sanity: value_pos should be non-zero
        print("\n=== VALUE_POS SANITY ===")
        print("train value_pos sum:", float(train_scored["value_pos"].sum()))
        print("val value_pos sum:", float(val_scored["value_pos"].sum()))
        print("test value_pos sum:", float(test_scored["value_pos"].sum()))
        print(
            "test total VaR (all churners):",
            float(test_scored.loc[test_scored["churn_90d"] == 1, "value_pos"].sum()),
        )

        # Evaluate VaR@K under policies
        val_report = evaluate_policies(val_scored, budgets)
        test_report = evaluate_policies(test_scored, budgets)

        # Save reports
        rep_dir = repo_root / "reports" / "evaluation"
        rep_dir.mkdir(parents=True, exist_ok=True)
        val_path = rep_dir / "val_results.csv"
        test_path = rep_dir / "test_results.csv"
        val_report.to_csv(val_path, index=False)
        test_report.to_csv(test_path, index=False)

        # Log reports as artifacts
        mlflow.log_artifact(str(val_path), artifact_path="reports")
        mlflow.log_artifact(str(test_path), artifact_path="reports")

        # Log key metrics for ML policy
        for split_name, rep in [("val", val_report), ("test", test_report)]:
            for k in budgets:
                row = rep[(rep["policy"] == "policy_ml") & (rep["budget_k"] == k)]
                if len(row) == 1:
                    step = int(float(k) * 100)
                    mlflow.log_metric(
                        f"{split_name}_policy_ml_value_at_risk",
                        float(row["value_at_risk"].iloc[0]),
                        step=step,
                    )
                    mlflow.log_metric(
                        f"{split_name}_policy_ml_var_frac",
                        float(row["var_covered_frac"].iloc[0]),
                        step=step,
                    )

        # Log uplift vs best heuristic
        _log_uplift(val_report, "val", budgets)
        _log_uplift(test_report, "test", budgets)

        # Write and log "promoted model" pointer
        prom_path = write_promotion_record(repo_root, run_id, model_name="churn_xgb_v1")
        mlflow.log_artifact(str(prom_path), artifact_path="promotion")
        print("\n=== PROMOTED MODEL POINTER WRITTEN ===")
        print(prom_path)

        # Print run summary
        print("\n=== MLFLOW RUN ===")
        print("tracking_uri:", tracking_uri)
        print("experiment:", mlflow_cfg.get("experiment_name", "churnxgb"))
        print("run_id:", run_id)
        print("data_version:", data_version)

        print("\n=== SPLIT SIZES ===")
        print(
            "train:",
            train_scored.shape,
            "val:",
            val_scored.shape,
            "test:",
            test_scored.shape,
        )

        print("\n=== FEATURE COLS COUNT ===")
        print(len(feature_cols))
        print("feature_cols:", feature_cols)

        print("\n=== VALIDATION REPORT (top rows) ===")
        print(val_report.head(20))

        print("\n=== TEST REPORT (top rows) ===")
        print(test_report.head(20))


if __name__ == "__main__":
    main()
