from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st
import yaml

from churnxgb.artifacts import ArtifactPaths
from churnxgb.modeling.interpretability import explain_prediction_rows
from churnxgb.pipeline.score import load_model, simulate_policy_by_budget
from churnxgb.policy.scoring import get_decision_policy_config
from churnxgb.evaluation.experiment_simulation import (
    get_experiment_config,
    simulate_experiment_by_budget,
)


ROOT = Path(__file__).resolve().parents[1]


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def _load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def _load_model_info() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    tracking_uri = "file:./mlruns_store"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", tracking_uri)

    return load_model(ROOT, tracking_uri)


@st.cache_data
def _load_app_config() -> dict:
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        return {"decision": get_decision_policy_config({})}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_df(path: Path) -> pd.DataFrame | None:
    return _load_csv(path) if path.exists() else None


def _artifact_status(label: str, exists: bool) -> str:
    return f"{'OK' if exists else 'Missing'} - {label}"


def main() -> None:
    st.set_page_config(page_title="ChurnXGB Dashboard", layout="wide")
    st.title("ChurnXGB Portfolio Dashboard")
    st.caption("Leakage-aware churn targeting with budget-constrained policy evaluation")
    runtime_root = ArtifactPaths.for_repo(ROOT).runtime_root
    reports = runtime_root / "reports"
    eval_dir = reports / "evaluation"
    figures = reports / "figures"
    outputs = runtime_root / "outputs"

    comparison = _safe_df(reports / "model_comparison.csv")
    predictions_path = outputs / "predictions" / "predictions_all.parquet"
    preds = _load_parquet(predictions_path) if predictions_path.exists() else None
    drift_path = reports / "monitoring" / "drift_latest.json"
    drift = _load_json(drift_path) if drift_path.exists() else None
    drift_history = _safe_df(reports / "monitoring" / "drift_history.csv")
    fi = _safe_df(reports / "feature_importance.csv")
    manifest_path = reports / "training_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    app_cfg = _load_app_config()
    decision_cfg = get_decision_policy_config(app_cfg)
    experiment_cfg = get_experiment_config(app_cfg)

    if comparison is None or preds is None:
        st.error("Missing required artifacts. Run the training and scoring pipelines first.")
        return

    best_model = manifest.get("best_model", comparison.iloc[0]["model"])
    best_row = comparison[comparison["model"] == best_model].iloc[0]
    available_budgets = [5, 10, 20]

    st.sidebar.header("Navigation")
    view = st.sidebar.radio(
        "Choose a view",
        [
            "Executive Summary",
            "Policy Simulator",
            "Model Performance",
            "Explainability",
            "Customer Explanation",
            "Customer Risk Explorer",
            "Experiment Simulation",
            "Drift Monitoring",
        ],
    )
    st.sidebar.header("Artifacts")
    st.sidebar.caption(_artifact_status("Model comparison", comparison is not None))
    st.sidebar.caption(_artifact_status("Predictions", preds is not None))
    st.sidebar.caption(_artifact_status("Feature importance", fi is not None))
    st.sidebar.caption(_artifact_status("Drift report", drift is not None))
    st.sidebar.caption(_artifact_status("Drift history", drift_history is not None))

    policy_path = eval_dir / f"{best_model}_test_policy_results.csv"
    policy_df = _load_csv(policy_path) if policy_path.exists() else None

    if view == "Executive Summary":
        st.header("Executive Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows Scored", f"{len(preds):,}")
        c2.metric("Best Model", str(best_model))
        c3.metric("Best Val VaR@10%", f"{best_row['val_value_at_risk']:,.0f}")
        c4.metric("Best Test ROC-AUC", f"{best_row['test_roc_auc']:.3f}")
        st.caption(
            "Use the sidebar to move between business summary, targeting simulation, customer action lists, and drift monitoring."
        )
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

    elif view == "Policy Simulator":
        st.header("Policy Simulator")
        st.caption(
            "Assumption-driven decision threshold analysis. This view applies flat configured intervention cost and assumed retention success to existing model scores. It is not causal inference."
        )
        budget_pct = st.slider("Target budget (%)", min_value=5, max_value=20, value=10, step=5)
        budget = budget_pct / 100.0
        needed_cols = {"policy_net_benefit", "expected_retained_value", "expected_cost", "churn_90d"}
        if not needed_cols.issubset(preds.columns):
            st.warning(
                "Scored prediction artifacts do not yet include decision-simulation columns. Re-run the scoring pipeline to populate them."
            )
            return
        if policy_df is not None:
            budget_row = policy_df[policy_df["budget_k"] == budget].sort_values("value_at_risk", ascending=False)
            top = preds.sort_values("policy_ml", ascending=False).head(max(1, round(len(preds) * budget)))
            sim_rows = simulate_policy_by_budget(preds, [budget])
            sim_row = sim_rows[0]
            pc1, pc2, pc3, pc4 = st.columns(4)
            ml_row = budget_row[budget_row["policy"] == "policy_ml"].iloc[0]
            rec_row = budget_row[budget_row["policy"] == "policy_recency"].iloc[0]
            rfm_row = budget_row[budget_row["policy"] == "policy_rfm"].iloc[0]
            pc1.metric("Captured VaR", f"{ml_row['value_at_risk']:,.0f}")
            pc2.metric("Net Benefit@K", f"{sim_row['comparison_net_benefit_at_k']:,.2f}")
            pc3.metric("Rank Changed", "Yes" if sim_row["ranking_changed"] else "No")
            pc4.metric("Selected Diff", f"{sim_row['n_selected_only_baseline'] + sim_row['n_selected_only_comparison']}")
            st.json(
                {
                    "decision_simulation_assumption_driven": True,
                    "decision_config": decision_cfg,
                    "comparison_note": "With flat configured economics, policy_net_benefit acts as a profitability threshold on the same underlying ranking.",
                }
            )
            compare_df = pd.DataFrame(
                [
                    {
                        "metric": "Value at Risk",
                        "policy_ml": sim_row["value_at_risk_baseline"],
                        "policy_net_benefit": sim_row["value_at_risk_comparison"],
                    },
                    {
                        "metric": "Net Benefit@K",
                        "policy_ml": sim_row["baseline_net_benefit_at_k"],
                        "policy_net_benefit": sim_row["comparison_net_benefit_at_k"],
                    },
                ]
            ).set_index("metric")
            st.bar_chart(compare_df, use_container_width=True)
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("% Customers With Rank Change", f"{sim_row['pct_customers_rank_changed']:.1%}")
            rc2.metric("Only In VaR Top-K", f"{sim_row['n_selected_only_baseline']}")
            rc3.metric("Only In Net-Benefit Top-K", f"{sim_row['n_selected_only_comparison']}")
            differing_ids = pd.DataFrame(
                {
                    "selected_only_policy_ml": pd.Series(sim_row["selected_only_baseline_customer_ids"]),
                    "selected_only_policy_net_benefit": pd.Series(sim_row["selected_only_comparison_customer_ids"]),
                }
            )
            if len(differing_ids.dropna(how="all")) > 0:
                st.subheader("Different Selected Customers")
                st.dataframe(differing_ids, use_container_width=True)
            st.bar_chart(
                budget_row.set_index("policy")[["value_at_risk", "precision_at_k", "lift_at_k"]],
                use_container_width=True,
            )
            st.caption(
                f"At a {budget_pct}% budget, the dashboard ranks {len(top):,} customers for outreach using the promoted model."
            )
        else:
            st.warning("Policy artifacts are missing for the promoted model.")

    elif view == "Model Performance":
        st.header("Model Performance")
        curve_cols = st.columns(2)
        roc_path = figures / "test_roc_curve.png"
        pr_path = figures / "test_pr_curve.png"
        lift_path = figures / "test_lift_curve.png"
        calib_path = figures / "test_calibration_curve.png"
        if roc_path.exists():
            curve_cols[0].image(str(roc_path), caption="ROC Curve")
        if pr_path.exists():
            curve_cols[1].image(str(pr_path), caption="Precision-Recall Curve")
        curve_cols2 = st.columns(2)
        if lift_path.exists():
            curve_cols2[0].image(str(lift_path), caption="Lift Curve")
        if calib_path.exists():
            curve_cols2[1].image(str(calib_path), caption="Calibration Curve")

    elif view == "Explainability":
        st.header("Explainability")
        if fi is not None:
            st.dataframe(fi.head(15), use_container_width=True)
        shap_path = figures / "shap_summary_bar.png"
        fallback_path = figures / "feature_importance.png"
        if shap_path.exists():
            st.image(str(shap_path), caption="SHAP Summary")
        elif fallback_path.exists():
            st.image(str(fallback_path), caption="Feature Importance")

    elif view == "Customer Explanation":
        st.header("Customer Explanation")
        st.caption(
            "Row-level explanations use the current promoted model. For the promoted logistic-regression pipeline, contributions are exact logit-space feature contributions relative to the standardized baseline, which corresponds to the training-data mean after scaling."
        )
        model_info = _load_model_info()
        feature_cols = model_info["feature_cols"]

        ranked = preds.sort_values("policy_ml", ascending=False).reset_index(drop=True)
        max_index = max(0, len(ranked) - 1)
        selected_rank = st.number_input(
            "Ranked customer row",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1,
        )
        selected = ranked.iloc[[int(selected_rank)]].copy()
        explanation = explain_prediction_rows(
            model=model_info["model"],
            X=selected[feature_cols],
            feature_cols=feature_cols,
            top_n=5,
        )[0]

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("CustomerID", str(selected.iloc[0]["CustomerID"]))
        ec2.metric("Churn Probability", f"{float(selected.iloc[0]['churn_prob']):.3f}")
        ec3.metric("Policy Score", f"{float(selected.iloc[0]['policy_ml']):.2f}")
        st.json(
            {
                "explanation_method": explanation["explanation_method"],
                "base_value_probability": explanation.get("base_value_probability"),
                "prediction_probability": explanation["prediction_probability"],
            }
        )
        pos_df = pd.DataFrame(explanation["top_positive_contributors"])
        neg_df = pd.DataFrame(explanation["top_negative_contributors"])
        if len(pos_df):
            st.subheader("Top Positive Contributors")
            st.dataframe(pos_df, use_container_width=True)
        if len(neg_df):
            st.subheader("Top Negative Contributors")
            st.dataframe(neg_df, use_container_width=True)

    elif view == "Customer Risk Explorer":
        st.header("Customer Risk Explorer")
        budget_option = st.selectbox("Target list to review", available_budgets, index=1)
        target_col = f"target_k{budget_option:02d}"
        targeted_only = st.checkbox("Show only customers currently targeted", value=True)
        customer_filter = st.text_input("Filter by CustomerID", "")
        row_limit = st.slider("Rows to display", min_value=25, max_value=500, value=100, step=25)

        display_df = preds[
            [
                "CustomerID",
                "invoice_month",
                "churn_prob",
                "value_pos",
                "policy_ml",
                "target_k05",
                "target_k10",
                "target_k20",
            ]
        ].sort_values("policy_ml", ascending=False)

        if targeted_only:
            display_df = display_df[display_df[target_col] == 1]
        if customer_filter.strip():
            display_df = display_df[
                display_df["CustomerID"].astype(str).str.contains(customer_filter.strip(), na=False)
            ]

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Customers Shown", f"{len(display_df):,}")
        rc2.metric("Selected Target List", f"Top {budget_option}%")
        rc3.metric("Avg Churn Probability", f"{display_df['churn_prob'].mean():.3f}" if len(display_df) else "n/a")
        st.dataframe(display_df.head(row_limit), use_container_width=True)

    elif view == "Experiment Simulation":
        st.header("Experiment Simulation")
        st.caption(
            "Deterministic business-case simulation on saved scored outputs. This is not causal inference, not observed uplift estimation, and does not report statistical confidence intervals."
        )
        needed_cols = {
            "policy_net_benefit",
            "assumed_success_rate_customer",
            "expected_retained_value",
            "value_pos",
            "churn_prob",
        }
        if not needed_cols.issubset(preds.columns):
            st.warning(
                "Scored prediction artifacts do not yet include the fields required for experiment simulation. Re-run the scoring pipeline to populate them."
            )
            return

        budget_pct = st.slider(
            "Experiment budget (%)", min_value=5, max_value=20, value=10, step=5, key="experiment_budget"
        )
        budget = budget_pct / 100.0
        result = simulate_experiment_by_budget(preds, [budget], experiment_cfg)[0]

        ex1, ex2, ex3, ex4 = st.columns(4)
        ex1.metric("Targeted Customers", f"{result['targeted_customers']:,}")
        ex2.metric("Treatment Customers", f"{result['treatment_customers']:,}")
        ex3.metric("Control Customers", f"{result['control_customers']:,}")
        ex4.metric("Incremental Retained Value", f"{result['incremental_retained_value']:,.2f}")

        ex5, ex6 = st.columns(2)
        ex5.metric(
            "Avg Incremental Value / Treated",
            f"{result['average_incremental_retained_value_per_treated_customer']:.2f}",
        )
        ex6.metric(
            "Avg Uplift Probability",
            f"{result['average_uplift_probability_treatment']:.3f}",
        )

        st.json(
            {
                "assumption_driven": True,
                "experiment_config": experiment_cfg,
                "limitation_note": result["limitation_note"],
            }
        )

        arms_df = pd.DataFrame(
            {
                "treatment_customer_ids": pd.Series(result["top_treated_customer_ids"]),
                "control_customer_ids": pd.Series(result["top_control_customer_ids"]),
            }
        )
        st.subheader("Sample Simulated Treatment / Control Customers")
        st.dataframe(arms_df, use_container_width=True)

    elif view == "Drift Monitoring":
        st.header("Drift Monitoring")
        if drift is not None:
            summary = drift.get("summary", {})
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Features OK", int(summary.get("n_ok", 0)))
            dc2.metric("Warnings", int(summary.get("n_warn", 0)))
            dc3.metric("Alerts", int(summary.get("n_alert", 0)))
            alerts = drift.get("alerts", {})
            if len(alerts):
                st.json(
                    {
                        "generated_at_utc": drift.get("generated_at_utc"),
                        "overall_status": alerts.get("overall_status"),
                        "n_warn_features": alerts.get("n_warn_features"),
                        "n_alert_features": alerts.get("n_alert_features"),
                    }
                )
            drift_features = pd.DataFrame(
                [
                    {"feature": key, **value}
                    for key, value in drift.get("features", {}).items()
                ]
            ).sort_values("psi", ascending=False)
            st.dataframe(drift_features, use_container_width=True)
            st.subheader("Current Score Distribution")
            st.json(drift.get("score_current_stats", {}))
            if drift_history is not None and len(drift_history):
                st.subheader("Drift History")
                hist_cols = [
                    col
                    for col in [
                        "generated_at_utc",
                        "overall_status",
                        "n_warn",
                        "n_alert",
                        "top_alert_feature",
                        "top_psi",
                        "score_mean",
                    ]
                    if col in drift_history.columns
                ]
                trend_cols = [col for col in ["top_psi", "score_mean"] if col in drift_history.columns]
                if trend_cols:
                    st.line_chart(
                        drift_history.sort_values("generated_at_utc").set_index("generated_at_utc")[trend_cols],
                        use_container_width=True,
                    )
                st.dataframe(
                    drift_history.sort_values("generated_at_utc", ascending=False)[hist_cols],
                    use_container_width=True,
                )
            else:
                st.caption("Drift history has not been created yet. Re-run scoring to append monitoring history.")
        else:
            st.warning("Drift artifact is missing.")


if __name__ == "__main__":
    main()
