from __future__ import annotations

from pathlib import Path
import json
import yaml
import pandas as pd

import mlflow

from churnxgb.modeling.model_utils import load_model_artifacts
from churnxgb.modeling.mlflow_loader import (
    load_promoted_sklearn_model_from_run_id,
    predict_proba_1,
)
from churnxgb.baselines.heuristics import add_heuristics
from churnxgb.policy.scoring import add_policy_scores
from churnxgb.monitoring.drift import drift_report, top_psi_features
from churnxgb.utils.hashing import sha256_file


def _resolve_tracking_uri(repo_root: Path, tracking_uri: str) -> str:
    if tracking_uri.startswith("file:./") or tracking_uri == "file:mlruns":
        abs_path = (repo_root / "mlruns").resolve()
        return "file:" + str(abs_path).replace("\\", "/")
    return tracking_uri


def _write_targets(
    df: pd.DataFrame, out_dir: Path, split_name: str, budgets: list[float]
) -> None:
    for k in budgets:
        n = max(1, int(round(len(df) * float(k))))
        top = df.sort_values("policy_ml", ascending=False).head(n).copy()
        top = top[
            [
                "CustomerID",
                "invoice_month",
                "T",
                "churn_prob",
                "value_pos",
                "policy_ml",
            ]
        ]
        out_path = out_dir / f"targets_{split_name}_k{int(k * 100):02d}.parquet"
        top.to_parquet(out_path, index=False)


def _resolve_promotion(repo_root: Path) -> dict | None:
    promoted_path = repo_root / "models" / "promoted" / "production.json"
    if promoted_path.exists():
        with open(promoted_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    budgets = [float(x) for x in cfg["eval"]["budgets"]]

    # Load features
    feats_path = repo_root / "data" / "processed" / "customer_month_features.parquet"
    df = pd.read_parquet(feats_path)
    data_version = sha256_file(feats_path)

    # Promotion record (if present)
    prom = _resolve_promotion(repo_root)

    # Always load feature columns from local registry (schema contract)
    model_name = prom.get("model_name", "churn_xgb_v1") if prom else "churn_xgb_v1"
    _local_model, feature_cols, _meta = load_model_artifacts(
        repo_root, model_name=model_name
    )

    # MLflow setup (for logging drift artifacts)
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    tracking_uri = _resolve_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "churnxgb"))

    # If promoted run_id exists, load model from MLflow; otherwise fall back to local joblib registry
    if prom and prom.get("run_id"):
        run_id = prom["run_id"]
        print("Using promoted run_id:", run_id)
        print("Using MLflow tracking_uri:", tracking_uri)

        model = load_promoted_sklearn_model_from_run_id(
            repo_root, run_id=run_id, tracking_uri=tracking_uri
        )
        model_source = f"mlflow:runs:/{run_id}/model"
    else:
        print("No promotion record found; using local registry model:", model_name)
        model = _local_model
        run_id = None
        model_source = f"local_registry:{model_name}"

    print("Model source:", model_source)

    # Add baselines (optional downstream comparisons)
    df = add_heuristics(df)

    # Predict churn probability
    df = df.copy()
    X = df[feature_cols]

    if model_source.startswith("mlflow:"):
        df["churn_prob"] = predict_proba_1(model, X)
    else:
        df["churn_prob"] = model.predict_proba(X)[:, 1]

    # Add policies (defines value_pos from trailing rev_sum_90d)
    df = add_policy_scores(df)
    for k in budgets:
        n = max(1, int(round(len(df) * float(k))))
        top_idx = df.sort_values("policy_ml", ascending=False).head(n).index
        df[f"target_k{int(k * 100):02d}"] = df.index.isin(top_idx).astype(int)

    # Drift monitoring
    ref_path = repo_root / "reports" / "monitoring" / "reference_profile.json"
    mon_dir = repo_root / "reports" / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)

    drift_out_path = mon_dir / "drift_latest.json"
    drift_top = None
    drift_summary = None

    if ref_path.exists():
        report = drift_report(
            ref_path,
            df,
            feature_cols,
            psi_threshold_warn=0.1,
            psi_threshold_alert=0.25,
            include_score_col=None,
        )
        with open(drift_out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        drift_summary = report.get("summary")
        drift_top = top_psi_features(report, top_n=10)

        print("\n=== DRIFT REPORT WRITTEN ===")
        print(drift_out_path)
        print("drift_summary:", drift_summary)
        print("top_psi_features:", drift_top)
        print("score_reference_stats:", report.get("score_reference_stats"))
        print("score_current_stats:", report.get("score_current_stats"))
    else:
        print("\n=== DRIFT REPORT SKIPPED ===")
        print("reference_profile.json not found at:", ref_path)

    # Write predictions + targets
    pred_dir = repo_root / "outputs" / "predictions"
    targ_dir = repo_root / "outputs" / "targets"
    pred_dir.mkdir(parents=True, exist_ok=True)
    targ_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / "predictions_all.parquet"
    df.to_parquet(pred_path, index=False)
    _write_targets(df, targ_dir, "all", budgets)

    print("=== SCORED OUTPUTS WRITTEN ===")
    print("predictions:", pred_path)
    for k in budgets:
        print("targets:", targ_dir / f"targets_all_k{int(k * 100):02d}.parquet")

    # Log scoring artifacts + drift report to MLflow (attach to training run if available; else create a scoring run)
    if run_id:
        # attach artifacts to the *promoted* run by starting a nested run is not supported in this simple setup.
        # Instead, we create a separate scoring run that records the linkage to the promoted run_id.
        with mlflow.start_run(run_name="score_run") as score_run:
            mlflow.log_param("linked_model_run_id", run_id)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("model_source", model_source)

            mlflow.log_artifact(str(pred_path), artifact_path="scoring_outputs")
            for k in budgets:
                mlflow.log_artifact(
                    str(targ_dir / f"targets_all_k{int(k * 100):02d}.parquet"),
                    artifact_path="scoring_outputs",
                )

            if ref_path.exists():
                mlflow.log_artifact(str(drift_out_path), artifact_path="monitoring")
                # log a few summary metrics for quick scanning
                if drift_summary:
                    mlflow.log_metric(
                        "drift_n_warn", float(drift_summary.get("n_warn", 0))
                    )
                    mlflow.log_metric(
                        "drift_n_alert", float(drift_summary.get("n_alert", 0))
                    )
                if drift_top and len(drift_top) > 0:
                    mlflow.log_metric("drift_top_psi", float(drift_top[0]["psi"]))

            print("\n=== MLFLOW SCORE RUN ===")
            print("score_run_id:", score_run.info.run_id)
            print("linked_model_run_id:", run_id)
    else:
        print("\n=== MLFLOW SCORE LOGGING SKIPPED ===")
        print("No promoted run_id found; scoring artifacts not logged to MLflow.")


if __name__ == "__main__":
    main()
