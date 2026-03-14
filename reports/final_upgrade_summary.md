# Final Upgrade Summary

## What Changed

- Expanded evaluation beyond Value-at-Risk with ROC-AUC, PR-AUC, Brier score, Precision@K, Recall@K, Lift@K, and calibration curve data.
- Added saved machine-readable evaluation outputs in `reports/evaluation/` and saved plots in `reports/figures/`.
- Added multi-model comparison across:
  - XGBoost
  - Logistic Regression
  - LightGBM
- Added rolling expanding-window temporal backtesting across 9 chronological folds.
- Added interpretability artifacts:
  - `reports/feature_importance.csv`
  - `reports/feature_analysis.md`
  - `reports/figures/feature_importance.png`
- Added a Streamlit dashboard at `dashboard/app.py` that reads saved artifacts rather than retraining.
- Added reproducibility support with `requirements.txt`.
- Added targeted tests for temporal splitting, leakage-safe feature selection, feature schema, and metric sanity.

## What Was Successfully Rerun

- `python -m churnxgb.pipeline.build_features`
- `python -m churnxgb.pipeline.train`
- `python -m churnxgb.pipeline.score`
- `pytest -q`
- Streamlit dashboard startup smoke test via `streamlit run dashboard/app.py --server.headless true`

## Final Metrics By Model

### Holdout Validation / Test Comparison At 10% Budget

| model | val_value_at_risk | test_value_at_risk | val_roc_auc | test_roc_auc | val_pr_auc | test_pr_auc | val_brier_score | test_brier_score |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| logistic_regression | 97,853.45 | 165,919.06 | 0.7629 | 0.7335 | 0.5716 | 0.5819 | 0.1726 | 0.2016 |
| xgboost | 76,760.36 | 164,787.80 | 0.7456 | 0.7233 | 0.5358 | 0.5805 | 0.1799 | 0.2077 |
| lightgbm | 72,032.55 | 164,581.86 | 0.7458 | 0.7205 | 0.5359 | 0.5749 | 0.1805 | 0.2089 |

### Promoted Model Test Targeting Metrics

Promoted model: `logistic_regression`

| budget_k | value_at_risk | var_covered_frac | precision_at_k | recall_at_k | lift_at_k |
|:--|--:|--:|--:|--:|--:|
| 5% | 108,496.47 | 0.1738 | 0.4604 | 0.0607 | 1.2140 |
| 10% | 165,919.06 | 0.2658 | 0.4065 | 0.1072 | 1.0717 |
| 20% | 260,599.82 | 0.4175 | 0.3939 | 0.2078 | 1.0385 |

## Best Model By Business Metric

- Best validation business metric: `logistic_regression`
- Selection rule used: `val_value_at_risk_at_10`
- Promoted MLflow run id: `9e6b146aa5e54773a9e5edb1f37eeff1`

## Best Model By ML Metric

- On the current validation and test holdout splits, `logistic_regression` also had the best ROC-AUC, PR-AUC, and lowest Brier score among the compared models.

## Whether Calibration Improved

- A calibration curve is now generated and saved to `reports/figures/test_calibration_curve.png`.
- Among the compared models, `logistic_regression` had the lowest Brier score on both validation and test, which is the strongest calibration signal currently tracked in the repository.
- Improvement versus the pre-upgrade repository cannot be measured directly because Brier score and calibration outputs did not previously exist.

## Whether The Dashboard Works

- Yes. A smoke test confirmed the app starts successfully with Streamlit and serves at a local URL when launched.
- Entry point: `dashboard/app.py`

## Limitations Still Remaining

- There is still no delayed-label monitoring for scored outputs after labels mature in the future.
- There is still no CI pipeline or environment lockfile beyond `requirements.txt`.
- The dashboard is portfolio/demo oriented, not a production application.
- The score pipeline still writes one promoted model at a time rather than serving multiple champion/challenger models concurrently.
