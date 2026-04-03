# Final Results Summary

## Dataset Summary

- Unique customers in processed modeling table: 5,942
- Customer-month rows: 26,993
- Train rows: 6,904
- Validation rows: 1,952
- Test rows: 2,779

These split sizes were verified from `.runtime/data/processed/customer_month_features.parquet` using the configured temporal windows:
- train through `2010-06`
- validation `2010-07` to `2010-08`
- test `2010-09` to `2010-10`

## Best Model

Promoted model: `logistic_regression`

Why it won:
- It had the best validation business metric used for promotion: `val_value_at_risk_at_10 = 97,853.45`
- It also had the strongest holdout ML metrics among the compared models:
  - ROC-AUC: 0.7335
  - PR-AUC: 0.5819
  - Brier score: 0.2016

### Promoted Model Holdout Metrics

- ROC-AUC: 0.7335
- PR-AUC: 0.5819
- Brier score: 0.2016
- Precision@10%: 0.4065
- Recall@10%: 0.1072
- Lift@10%: 1.0717
- Value-at-Risk@10%: 165,919.06

Promotion metadata was verified in `.runtime/models/promoted/production.json`:
- run id: `b3fccc8ed96645b185b38ab1d024a970`
- registry artifact: `churn_logistic_regression_v1`

## Model Comparison Table

| model | ROC-AUC | PR-AUC | Brier | Recall@10 | VaR@10 |
|:--|--:|--:|--:|--:|--:|
| logistic_regression | 0.7335 | 0.5819 | 0.2016 | 0.1072 | 165,919.06 |
| xgboost | 0.7233 | 0.5805 | 0.2077 | 0.1053 | 164,787.80 |
| lightgbm | 0.7205 | 0.5749 | 0.2089 | 0.1063 | 164,581.86 |

## Backtesting Results

- Number of rolling temporal folds: 9
- Best-model mean ROC-AUC across folds: 0.7579
- Best-model ROC-AUC std across folds: 0.0265
- Best-model mean VaR@10 across folds: 155,696.69
- Best-model VaR@10 std across folds: 87,411.42

Interpretation:
- ROC-AUC is fairly stable across folds.
- VaR@10 varies substantially across time windows, which is expected because customer value at risk changes across cohorts and seasonality periods.

## Policy Impact

Current operating policy example: target the top 10% of customers ranked by:

`policy_ml = churn_probability * trailing_90d_value_proxy`

At 10% budget on the holdout test split:
- Customers targeted: 278 of 2,779
- Value-at-Risk captured: 165,919.06
- Fraction of total test value-at-risk captured: 26.58%
- Captured churners: 113

### Uplift vs Heuristic Baseline

Best heuristic baseline at 10% on the same holdout split:
- Recency policy VaR@10: 107,171.34

ML policy uplift at 10%:
- Absolute uplift: 58,747.72
- Relative uplift: 54.8%

### Assumptions

- This policy analysis compares ranking quality under a fixed targeting budget.
- It does **not** assume a realized treatment effect or retained revenue from outreach.
- Any real intervention ROI would require an explicit response-rate or treatment-lift assumption, which is not modeled here.

## Artifact Verification

The following regenerated artifacts were verified after the rerun:
- `reports/final_upgrade_summary.md`
- `.runtime/reports/model_eval_summary.md`
- `.runtime/reports/model_comparison.csv`
- `.runtime/reports/backtest_summary.csv`
- `.runtime/reports/feature_importance.csv`
- `.runtime/reports/figures/test_roc_curve.png`
- `.runtime/reports/figures/test_pr_curve.png`
- `.runtime/reports/figures/test_lift_curve.png`
- `.runtime/reports/figures/test_calibration_curve.png`
- `.runtime/reports/monitoring/drift_latest.json`
- `.runtime/outputs/predictions/predictions_all.parquet`
- `.runtime/outputs/targets/targets_all_k10.parquet`
- MLflow run directories for the latest training and scoring runs

## Dashboard Validation

The Streamlit dashboard launched successfully in headless mode and loaded saved artifacts without retraining.

Verified sections backed by current artifacts:
- Executive summary
- Policy simulator
- Model performance charts
- Explainability
- Customer risk explorer
- Drift monitoring
