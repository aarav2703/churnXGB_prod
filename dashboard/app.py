from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EVAL = REPORTS / "evaluation"
FIGURES = REPORTS / "figures"
OUTPUTS = ROOT / "outputs"


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


def _safe_df(path: Path) -> pd.DataFrame | None:
    return _load_csv(path) if path.exists() else None


def main() -> None:
    st.set_page_config(page_title="ChurnXGB Dashboard", layout="wide")
    st.title("ChurnXGB Portfolio Dashboard")
    st.caption("Leakage-aware churn targeting with budget-constrained policy evaluation")

    comparison = _safe_df(REPORTS / "model_comparison.csv")
    predictions_path = OUTPUTS / "predictions" / "predictions_all.parquet"
    preds = _load_parquet(predictions_path) if predictions_path.exists() else None
    drift_path = REPORTS / "monitoring" / "drift_latest.json"
    drift = _load_json(drift_path) if drift_path.exists() else None
    fi = _safe_df(REPORTS / "feature_importance.csv")
    manifest_path = REPORTS / "training_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}

    if comparison is None or preds is None:
        st.error("Missing required artifacts. Run the training and scoring pipelines first.")
        return

    best_model = manifest.get("best_model", comparison.iloc[0]["model"])
    best_row = comparison[comparison["model"] == best_model].iloc[0]

    st.header("Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Scored", f"{len(preds):,}")
    c2.metric("Best Model", str(best_model))
    c3.metric("Best Val VaR@10%", f"{best_row['val_value_at_risk']:,.0f}")
    c4.metric("Best Test ROC-AUC", f"{best_row['test_roc_auc']:.3f}")

    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True)

    st.header("Policy Simulator")
    budget_pct = st.slider("Target budget (%)", min_value=5, max_value=20, value=10, step=5)
    budget = budget_pct / 100.0
    policy_path = EVAL / f"{best_model}_test_policy_results.csv"
    policy_df = _load_csv(policy_path) if policy_path.exists() else None
    if policy_df is not None:
        budget_row = policy_df[policy_df["budget_k"] == budget].sort_values("value_at_risk", ascending=False)
        top = preds.sort_values("policy_ml", ascending=False).head(max(1, round(len(preds) * budget)))
        pc1, pc2, pc3, pc4 = st.columns(4)
        ml_row = budget_row[budget_row["policy"] == "policy_ml"].iloc[0]
        rec_row = budget_row[budget_row["policy"] == "policy_recency"].iloc[0]
        rfm_row = budget_row[budget_row["policy"] == "policy_rfm"].iloc[0]
        pc1.metric("Captured VaR", f"{ml_row['value_at_risk']:,.0f}")
        pc2.metric("Captured Churners", f"{int(ml_row['captured_churners'])}")
        pc3.metric("Precision@K", f"{ml_row['precision_at_k']:.3f}")
        pc4.metric("Uplift vs Best Heuristic", f"{(ml_row['value_at_risk'] / max(rec_row['value_at_risk'], rfm_row['value_at_risk']) - 1):.1%}")
        st.bar_chart(
            budget_row.set_index("policy")[["value_at_risk", "precision_at_k", "lift_at_k"]],
            use_container_width=True,
        )
        st.caption(f"Top-{budget_pct}% targeted customers: {len(top):,}")

    st.header("Model Performance")
    curve_cols = st.columns(2)
    roc_path = FIGURES / "test_roc_curve.png"
    pr_path = FIGURES / "test_pr_curve.png"
    lift_path = FIGURES / "test_lift_curve.png"
    calib_path = FIGURES / "test_calibration_curve.png"
    if roc_path.exists():
        curve_cols[0].image(str(roc_path), caption="ROC Curve")
    if pr_path.exists():
        curve_cols[1].image(str(pr_path), caption="Precision-Recall Curve")
    curve_cols2 = st.columns(2)
    if lift_path.exists():
        curve_cols2[0].image(str(lift_path), caption="Lift Curve")
    if calib_path.exists():
        curve_cols2[1].image(str(calib_path), caption="Calibration Curve")

    st.header("Explainability")
    if fi is not None:
        st.dataframe(fi.head(15), use_container_width=True)
    shap_path = FIGURES / "shap_summary_bar.png"
    fallback_path = FIGURES / "feature_importance.png"
    if shap_path.exists():
        st.image(str(shap_path), caption="SHAP Summary")
    elif fallback_path.exists():
        st.image(str(fallback_path), caption="Feature Importance")

    st.header("Customer Risk Explorer")
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
    st.dataframe(display_df.head(500), use_container_width=True)

    st.header("Drift Monitoring")
    if drift is not None:
        drift_features = pd.DataFrame(
            [
                {"feature": key, **value}
                for key, value in drift.get("features", {}).items()
            ]
        ).sort_values("psi", ascending=False)
        st.dataframe(drift_features, use_container_width=True)
        st.json(drift.get("score_current_stats", {}))


if __name__ == "__main__":
    main()
