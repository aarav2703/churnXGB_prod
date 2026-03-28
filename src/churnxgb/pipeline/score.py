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
from churnxgb.inference.contracts import (
    build_prediction_output,
    load_inference_contract,
    validate_inference_frame,
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
) -> dict[float, Path]:
    out_paths: dict[float, Path] = {}
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
        out_paths[float(k)] = out_path
    return out_paths


def _resolve_promotion(repo_root: Path) -> dict | None:
    promoted_path = repo_root / "models" / "promoted" / "production.json"
    if promoted_path.exists():
        with open(promoted_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model(repo_root: Path, tracking_uri: str) -> dict:
    prom = _resolve_promotion(repo_root)

    model_name = prom.get("model_name", "churn_xgb_v1") if prom else "churn_xgb_v1"
    local_model, feature_cols, _meta = load_model_artifacts(
        repo_root, model_name=model_name
    )
    contract = load_inference_contract(
        repo_root, model_name=model_name, feature_cols=feature_cols
    )

    requested_run_id = prom.get("run_id") if prom else None
    model = local_model
    run_id = None
    model_source = f"local_registry:{model_name}"
    fallback_reason = None

    if requested_run_id:
        try:
            model = load_promoted_sklearn_model_from_run_id(
                repo_root, run_id=requested_run_id, tracking_uri=tracking_uri
            )
            run_id = requested_run_id
            model_source = f"mlflow:runs:/{requested_run_id}/model"
        except Exception as exc:
            fallback_reason = str(exc)

    return {
        "model": model,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "contract": contract,
        "model_source": model_source,
        "run_id": run_id,
        "requested_run_id": requested_run_id,
        "fallback_reason": fallback_reason,
    }


def score_dataframe(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    contract: dict,
    budgets: list[float],
    model_source: str,
) -> pd.DataFrame:
    scored = add_heuristics(df.copy())

    X = scored[feature_cols]
    validate_inference_frame(X, contract["inference_input_columns"])

    if model_source.startswith("mlflow:"):
        scored["churn_prob"] = predict_proba_1(model, X)
    else:
        scored["churn_prob"] = model.predict_proba(X)[:, 1]

    scored = add_policy_scores(scored)
    for k in budgets:
        n = max(1, int(round(len(scored) * float(k))))
        top_idx = scored.sort_values("policy_ml", ascending=False).head(n).index
        scored[f"target_k{int(k * 100):02d}"] = scored.index.isin(top_idx).astype(int)

    return scored


def build_outputs(
    repo_root: Path,
    df: pd.DataFrame,
    feature_cols: list[str],
    budgets: list[float],
    split_name: str = "all",
) -> dict:
    ref_path = repo_root / "reports" / "monitoring" / "reference_profile.json"
    mon_dir = repo_root / "reports" / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)

    drift_out_path = mon_dir / "drift_latest.json"
    drift_report_payload = None
    drift_top = None
    drift_summary = None

    if ref_path.exists():
        drift_report_payload = drift_report(
            ref_path,
            df,
            feature_cols,
            psi_threshold_warn=0.1,
            psi_threshold_alert=0.25,
            include_score_col=None,
        )
        with open(drift_out_path, "w", encoding="utf-8") as f:
            json.dump(drift_report_payload, f, indent=2)

        drift_summary = drift_report_payload.get("summary")
        drift_top = top_psi_features(drift_report_payload, top_n=10)

    pred_dir = repo_root / "outputs" / "predictions"
    targ_dir = repo_root / "outputs" / "targets"
    pred_dir.mkdir(parents=True, exist_ok=True)
    targ_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"predictions_{split_name}.parquet"
    inference_pred_path = pred_dir / f"predictions_inference.parquet"
    df.to_parquet(pred_path, index=False)
    build_prediction_output(df).to_parquet(inference_pred_path, index=False)
    target_paths = _write_targets(df, targ_dir, split_name, budgets)

    return {
        "pred_path": pred_path,
        "inference_pred_path": inference_pred_path,
        "target_paths": target_paths,
        "ref_path": ref_path,
        "drift_out_path": drift_out_path,
        "drift_report": drift_report_payload,
        "drift_summary": drift_summary,
        "drift_top": drift_top,
    }


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

    # MLflow setup (for logging drift artifacts)
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    tracking_uri = _resolve_tracking_uri(repo_root, tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "churnxgb"))

    model_info = load_model(repo_root, tracking_uri)
    if model_info["requested_run_id"]:
        print("Using promoted run_id:", model_info["requested_run_id"])
        print("Using MLflow tracking_uri:", tracking_uri)
    else:
        print("No promotion record found; using local registry model:", model_info["model_name"])
    if model_info["fallback_reason"]:
        print("Falling back to local registry model after MLflow load failure:")
        print(model_info["fallback_reason"])
    print("Model source:", model_info["model_source"])

    scored_df = score_dataframe(
        df,
        model=model_info["model"],
        feature_cols=model_info["feature_cols"],
        contract=model_info["contract"],
        budgets=budgets,
        model_source=model_info["model_source"],
    )
    output_info = build_outputs(
        repo_root,
        scored_df,
        feature_cols=model_info["feature_cols"],
        budgets=budgets,
        split_name="all",
    )

    if output_info["drift_report"] is not None:
        print("\n=== DRIFT REPORT WRITTEN ===")
        print(output_info["drift_out_path"])
        print("drift_summary:", output_info["drift_summary"])
        print("top_psi_features:", output_info["drift_top"])
        print("score_reference_stats:", output_info["drift_report"].get("score_reference_stats"))
        print("score_current_stats:", output_info["drift_report"].get("score_current_stats"))
    else:
        print("\n=== DRIFT REPORT SKIPPED ===")
        print("reference_profile.json not found at:", output_info["ref_path"])

    print("=== SCORED OUTPUTS WRITTEN ===")
    print("predictions:", output_info["pred_path"])
    print("inference_predictions:", output_info["inference_pred_path"])
    for k in budgets:
        print("targets:", output_info["target_paths"][float(k)])

    # Log scoring artifacts + drift report to MLflow (attach to training run if available; else create a scoring run)
    if model_info["run_id"]:
        # attach artifacts to the *promoted* run by starting a nested run is not supported in this simple setup.
        # Instead, we create a separate scoring run that records the linkage to the promoted run_id.
        with mlflow.start_run(run_name="score_run") as score_run:
            mlflow.log_param("linked_model_run_id", model_info["run_id"])
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("model_source", model_info["model_source"])

            mlflow.log_artifact(str(output_info["pred_path"]), artifact_path="scoring_outputs")
            mlflow.log_artifact(str(output_info["inference_pred_path"]), artifact_path="scoring_outputs")
            for k in budgets:
                mlflow.log_artifact(
                    str(output_info["target_paths"][float(k)]),
                    artifact_path="scoring_outputs",
                )

            if output_info["drift_report"] is not None:
                mlflow.log_artifact(str(output_info["drift_out_path"]), artifact_path="monitoring")
                # log a few summary metrics for quick scanning
                if output_info["drift_summary"]:
                    mlflow.log_metric(
                        "drift_n_warn", float(output_info["drift_summary"].get("n_warn", 0))
                    )
                    mlflow.log_metric(
                        "drift_n_alert", float(output_info["drift_summary"].get("n_alert", 0))
                    )
                if output_info["drift_top"] and len(output_info["drift_top"]) > 0:
                    mlflow.log_metric("drift_top_psi", float(output_info["drift_top"][0]["psi"]))

            print("\n=== MLFLOW SCORE RUN ===")
            print("score_run_id:", score_run.info.run_id)
            print("linked_model_run_id:", model_info["run_id"])
    else:
        print("\n=== MLFLOW SCORE LOGGING SKIPPED ===")
        print("No promoted run_id found; scoring artifacts not logged to MLflow.")


if __name__ == "__main__":
    main()
