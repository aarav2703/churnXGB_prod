from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import uuid
import yaml
import pandas as pd

import mlflow

from churnxgb.modeling.model_utils import load_model_artifacts
from churnxgb.inference.contracts import (
    build_prediction_output,
    load_inference_contract,
    validate_inference_frame,
)
from churnxgb.baselines.heuristics import add_heuristics
from churnxgb.evaluation.metrics import net_benefit_comparison_at_k
from churnxgb.evaluation.report import evaluate_policies
from churnxgb.monitoring.alerts import (
    get_monitoring_alert_config,
    summarize_drift_alerts,
)
from churnxgb.monitoring.drift import drift_report, top_psi_features
from churnxgb.monitoring.drift import compute_decision_drift
from churnxgb.monitoring.history import build_drift_history_frame
from churnxgb.paths import resolve_runtime_root
from churnxgb.policy.scoring import (
    add_policy_scores,
    get_decision_policy_config,
    get_targeting_policy_name,
)
from churnxgb.utils.hashing import sha256_file
from churnxgb.utils.io import atomic_write_csv, atomic_write_json


def _resolve_tracking_uri(repo_root: Path, tracking_uri: str) -> str:
    if tracking_uri.startswith("file:./") or tracking_uri == "file:mlruns":
        abs_path = (repo_root / "mlruns_store").resolve()
        return "file:" + str(abs_path).replace("\\", "/")
    return tracking_uri


def _resolve_promotion(repo_root: Path) -> dict | None:
    runtime_root = resolve_runtime_root(repo_root)
    promoted_path = runtime_root / "models" / "promoted" / "production.json"
    if promoted_path.exists():
        with open(promoted_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model(repo_root: Path, tracking_uri: str) -> dict:
    prom = _resolve_promotion(repo_root)

    model_name = prom.get("model_name", "churn_xgb_v1") if prom else "churn_xgb_v1"
    requested_run_id = prom.get("run_id") if prom else None
    run_id = requested_run_id
    if prom:
        registry_path = Path(prom["registry_path"])
        if not registry_path.exists():
            raise RuntimeError(
                f"Promoted registry path does not exist: {registry_path}"
            )
        model_name = prom["model_name"]
        model, feature_cols, _meta = load_model_artifacts(repo_root, model_name=model_name)
        contract = load_inference_contract(
            repo_root, model_name=model_name, feature_cols=feature_cols
        )
        model_source = f"promoted_registry:{model_name}"
    else:
        model, feature_cols, _meta = load_model_artifacts(repo_root, model_name=model_name)
        contract = load_inference_contract(
            repo_root, model_name=model_name, feature_cols=feature_cols
        )
        model_source = f"local_registry:{model_name}"

    return {
        "model": model,
        "model_name": model_name,
        "feature_cols": feature_cols,
        "contract": contract,
        "model_source": model_source,
        "run_id": run_id,
        "requested_run_id": requested_run_id,
        "fallback_reason": None,
    }


def score_dataframe(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    contract: dict,
    budgets: list[float],
    model_source: str,
    decision_cfg: dict[str, float] | None = None,
) -> pd.DataFrame:
    scored = add_heuristics(df.copy())

    X = scored[feature_cols]
    validate_inference_frame(X, contract["inference_input_columns"])

    scored["churn_prob"] = model.predict_proba(X)[:, 1]

    scored = add_policy_scores(scored, decision_cfg=decision_cfg)
    ranking_policy = get_targeting_policy_name(decision_cfg)
    for k in budgets:
        n = max(1, int(round(len(scored) * float(k))))
        top_idx = scored.sort_values(ranking_policy, ascending=False).head(n).index
        scored[f"target_k{int(k * 100):02d}"] = scored.index.isin(top_idx).astype(int)

    return scored


def simulate_policy_by_budget(
    scored_df: pd.DataFrame,
    budgets: list[float],
    baseline_policy: str = "policy_ml",
    comparison_policy: str = "policy_net_benefit",
) -> list[dict]:
    """
    Assumption-driven decision simulation using already-scored rows.

    With constant intervention cost, the comparison policy can preserve the same
    ranking order as the baseline policy. This function still quantifies the
    simulated economics at each budget.
    """
    report_df = evaluate_policies(scored_df, budgets)
    baseline_sorted = scored_df.sort_values(baseline_policy, ascending=False).reset_index()
    comparison_sorted = scored_df.sort_values(comparison_policy, ascending=False).reset_index()
    baseline_ranks = {
        int(original_idx): int(rank)
        for rank, original_idx in enumerate(baseline_sorted["index"].tolist())
    }
    comparison_ranks = {
        int(original_idx): int(rank)
        for rank, original_idx in enumerate(comparison_sorted["index"].tolist())
    }
    n_rank_changed = sum(
        1
        for idx in baseline_ranks
        if baseline_ranks[idx] != comparison_ranks.get(idx)
    )
    pct_rank_changed = (n_rank_changed / len(baseline_ranks)) if baseline_ranks else 0.0

    rows: list[dict] = []
    for k in budgets:
        comparison = net_benefit_comparison_at_k(
            scored_df,
            baseline_col=baseline_policy,
            comparison_col=comparison_policy,
            k=k,
        )
        baseline_row = report_df[
            (report_df["policy"] == baseline_policy) & (report_df["budget_k"] == k)
        ]
        comparison_row = report_df[
            (report_df["policy"] == comparison_policy) & (report_df["budget_k"] == k)
        ]
        top_n = max(1, int(round(len(scored_df) * float(k))))
        baseline_top = scored_df.sort_values(baseline_policy, ascending=False).head(top_n)
        comparison_top = scored_df.sort_values(comparison_policy, ascending=False).head(top_n)
        baseline_ids = set(baseline_top["CustomerID"].astype(str)) if "CustomerID" in scored_df.columns else set()
        comparison_ids = set(comparison_top["CustomerID"].astype(str)) if "CustomerID" in scored_df.columns else set()
        only_baseline = sorted(baseline_ids - comparison_ids)
        only_comparison = sorted(comparison_ids - baseline_ids)
        overlap = len(baseline_ids & comparison_ids)
        overlap_pct = (overlap / top_n) if top_n > 0 else 0.0
        selected_customers_differ = bool(len(only_baseline) > 0 or len(only_comparison) > 0)
        rows.append(
            {
                "budget_k": float(k),
                "assumption_driven": True,
                "baseline_policy": baseline_policy,
                "comparison_policy": comparison_policy,
                "ranking_changed": selected_customers_differ,
                "n_customers_rank_changed": int(n_rank_changed),
                "pct_customers_rank_changed": float(pct_rank_changed),
                "value_at_risk_baseline": float(baseline_row["value_at_risk"].iloc[0])
                if len(baseline_row) == 1
                else None,
                "value_at_risk_comparison": float(comparison_row["value_at_risk"].iloc[0])
                if len(comparison_row) == 1
                else None,
                "baseline_net_benefit_at_k": comparison["baseline_net_benefit_at_k"],
                "comparison_net_benefit_at_k": comparison["comparison_net_benefit_at_k"],
                "comparison_minus_baseline": comparison["comparison_minus_baseline"],
                "selected_customers_differ": selected_customers_differ,
                "selection_overlap_at_k": float(overlap_pct),
                "n_selected_only_baseline": int(len(only_baseline)),
                "n_selected_only_comparison": int(len(only_comparison)),
                "selected_only_baseline_customer_ids": only_baseline[:20],
                "selected_only_comparison_customer_ids": only_comparison[:20],
            }
        )
    return rows


def build_outputs(
    repo_root: Path,
    df: pd.DataFrame,
    feature_cols: list[str],
    budgets: list[float],
    split_name: str = "all",
    monitoring_cfg: dict[str, float] | None = None,
    decision_cfg: dict | None = None,
) -> dict:
    runtime_root = resolve_runtime_root(repo_root)
    ref_path = runtime_root / "reports" / "monitoring" / "reference_profile.json"
    mon_dir = runtime_root / "reports" / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = runtime_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    drift_out_path = mon_dir / "drift_latest.json"
    drift_history_path = mon_dir / "drift_history.csv"
    decision_drift_path = reports_dir / "decision_drift.csv"
    drift_report_payload = None
    drift_top = None
    drift_summary = None
    drift_alerts = None
    generated_at_utc = pd.Timestamp.utcnow().isoformat()
    alert_cfg = monitoring_cfg or get_monitoring_alert_config({})
    ranking_policy = get_targeting_policy_name(
        {"targeting_policy": "policy_net_benefit", **(decision_cfg or {})}
        if decision_cfg is not None
        else None
    )
    if "decision_targeting_policy" not in df.columns:
        df = df.copy()
        df["decision_targeting_policy"] = ranking_policy
    decision_drift_df = compute_decision_drift(df, budgets, ranking_policy)

    if ref_path.exists():
        drift_report_payload = drift_report(
            ref_path,
            df,
            feature_cols,
            psi_threshold_warn=float(alert_cfg["warn_threshold"]),
            psi_threshold_alert=float(alert_cfg["alert_threshold"]),
            include_score_col=None,
        )
        drift_alerts = summarize_drift_alerts(drift_report_payload)
        drift_report_payload["generated_at_utc"] = generated_at_utc
        drift_report_payload["alerts"] = drift_alerts
        drift_summary = drift_report_payload.get("summary")
        drift_top = top_psi_features(drift_report_payload, top_n=10)
        drift_history_df = build_drift_history_frame(
            drift_history_path,
            drift_report_payload,
            drift_alerts,
            generated_at_utc,
        )
    else:
        drift_history_df = None

    pred_dir = runtime_root / "outputs" / "predictions"
    targ_dir = runtime_root / "outputs" / "targets"
    pred_dir.mkdir(parents=True, exist_ok=True)
    targ_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pred_dir / f"predictions_{split_name}.parquet"
    inference_pred_path = pred_dir / f"predictions_inference.parquet"
    stage_dir = runtime_root / ".staging" / f"score_{uuid.uuid4().hex}"
    stage_pred_dir = stage_dir / "predictions"
    stage_targ_dir = stage_dir / "targets"
    stage_mon_dir = stage_dir / "monitoring"
    stage_reports_dir = stage_dir / "reports"
    stage_pred_dir.mkdir(parents=True, exist_ok=True)
    stage_targ_dir.mkdir(parents=True, exist_ok=True)
    stage_mon_dir.mkdir(parents=True, exist_ok=True)
    stage_reports_dir.mkdir(parents=True, exist_ok=True)

    stage_pred_path = stage_pred_dir / pred_path.name
    stage_inference_pred_path = stage_pred_dir / inference_pred_path.name
    df.to_parquet(stage_pred_path, index=False)
    build_prediction_output(df).to_parquet(stage_inference_pred_path, index=False)

    staged_target_paths: dict[float, Path] = {}
    for k in budgets:
        stage_target_path = stage_targ_dir / f"targets_{split_name}_k{int(k * 100):02d}.parquet"
        n = max(1, int(round(len(df) * float(k))))
        top = df.sort_values(ranking_policy, ascending=False).head(n).copy()
        top = top[
            [
                "CustomerID",
                "invoice_month",
                "T",
                "churn_prob",
                "value_pos",
                "policy_ml",
                "decision_targeting_policy",
                "assumed_success_rate_customer",
                "intervention_cost_customer",
                "expected_retained_value",
                "expected_cost",
                "policy_net_benefit",
            ]
        ]
        top.to_parquet(stage_target_path, index=False)
        staged_target_paths[float(k)] = stage_target_path

    stage_drift_path = stage_mon_dir / drift_out_path.name
    stage_history_path = stage_mon_dir / drift_history_path.name
    if drift_report_payload is not None:
        atomic_write_json(stage_drift_path, drift_report_payload)
        atomic_write_csv(drift_history_df, stage_history_path, index=False)
    stage_decision_drift_path = stage_reports_dir / decision_drift_path.name
    atomic_write_csv(decision_drift_df, stage_decision_drift_path, index=False)

    try:
        os.replace(stage_pred_path, pred_path)
        os.replace(stage_inference_pred_path, inference_pred_path)
        target_paths: dict[float, Path] = {}
        for k, staged in staged_target_paths.items():
            final_path = targ_dir / staged.name
            os.replace(staged, final_path)
            target_paths[float(k)] = final_path
        if drift_report_payload is not None:
            os.replace(stage_drift_path, drift_out_path)
            os.replace(stage_history_path, drift_history_path)
        os.replace(stage_decision_drift_path, decision_drift_path)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    return {
        "pred_path": pred_path,
        "inference_pred_path": inference_pred_path,
        "target_paths": target_paths,
        "ref_path": ref_path,
        "drift_out_path": drift_out_path,
        "drift_history_path": drift_history_path,
        "decision_drift_path": decision_drift_path,
        "drift_report": drift_report_payload,
        "drift_summary": drift_summary,
        "drift_top": drift_top,
        "drift_alerts": drift_alerts,
        "decision_drift": decision_drift_df,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cfg_path = repo_root / "config" / "config.yaml"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    runtime_root = resolve_runtime_root(repo_root, cfg)

    budgets = [float(x) for x in cfg["eval"]["budgets"]]
    decision_cfg = get_decision_policy_config(cfg)
    monitoring_cfg = get_monitoring_alert_config(cfg)

    # Load features
    feats_path = runtime_root / "data" / "processed" / "customer_month_features.parquet"
    df = pd.read_parquet(feats_path)
    data_version = sha256_file(feats_path)

    # MLflow setup (for logging drift artifacts)
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "file:./mlruns_store")
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
        decision_cfg=decision_cfg,
    )
    output_info = build_outputs(
        repo_root,
        scored_df,
        feature_cols=model_info["feature_cols"],
        budgets=budgets,
        split_name="all",
        monitoring_cfg=monitoring_cfg,
        decision_cfg=decision_cfg,
    )

    if output_info["drift_report"] is not None:
        print("\n=== DRIFT REPORT WRITTEN ===")
        print(output_info["drift_out_path"])
        print("drift_summary:", output_info["drift_summary"])
        print("top_psi_features:", output_info["drift_top"])
        print("score_reference_stats:", output_info["drift_report"].get("score_reference_stats"))
        print("score_current_stats:", output_info["drift_report"].get("score_current_stats"))
        print("drift_alerts:", output_info["drift_alerts"])
        print("drift_history:", output_info["drift_history_path"])
    else:
        print("\n=== DRIFT REPORT SKIPPED ===")
        print("reference_profile.json not found at:", output_info["ref_path"])

    print("=== SCORED OUTPUTS WRITTEN ===")
    print("predictions:", output_info["pred_path"])
    print("inference_predictions:", output_info["inference_pred_path"])
    print("decision_drift:", output_info["decision_drift_path"])
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
            mlflow.log_artifact(
                str(output_info["decision_drift_path"]),
                artifact_path="monitoring",
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
