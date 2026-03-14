# Improvement Plan

## Current Verified Capabilities

- Raw data ingestion loads `data/raw/online_retail_II.csv` through `src/churnxgb/data/load.py`.
- Cleaning, invoice aggregation, event construction, customer-month assembly, labeling, and feature generation run through `src/churnxgb/pipeline/build_features.py`.
- Churn labeling is point-in-time aware and implemented in `src/churnxgb/labeling/churn_90d.py`.
- Training currently uses a temporal split, heuristic baselines, one XGBoost classifier, MLflow logging, and promotion metadata in `src/churnxgb/pipeline/train.py`.
- Business-aware policy scoring is implemented in `src/churnxgb/policy/scoring.py`.
- Current evaluation computes Value-at-Risk style metrics at 5%, 10%, and 20% budgets in `src/churnxgb/evaluation/metrics.py` and `src/churnxgb/evaluation/report.py`.
- Scoring writes predictions to `outputs/predictions/predictions_all.parquet` and target lists to `outputs/targets/`.
- Monitoring writes a drift reference profile and PSI-based drift report in `reports/monitoring/`.
- The runtime environment already has `sklearn`, `xgboost`, `lightgbm`, `mlflow`, `matplotlib`, `streamlit`, `shap`, and `pytest` available.

## Current Verified Limitations

- Evaluation is narrow: no ROC-AUC, PR-AUC, Brier score, calibration analysis, precision@K, recall@K, or lift/gains outputs are implemented.
- There is only one learned model family in the training pipeline.
- There is no rolling temporal backtest summary across multiple folds.
- There are no interpretability artifacts in `reports/`.
- There is no dashboard app.
- `tests/` is empty.
- `pyproject.toml` does not declare the actual runtime dependencies used by the project.
- README mentions ROC-AUC as a secondary check, but that metric is not currently implemented in code. This will be corrected.

## Exact Changes To Make

1. Extend evaluation with:
   - ROC-AUC
   - PR-AUC
   - Brier score
   - Precision@K
   - Recall@K
   - Lift@K
   - calibration curve data
   - saved plots and markdown summaries
2. Add rolling-origin temporal backtesting over multiple month-based folds using the canonical point-in-time feature table.
3. Add model comparison for:
   - XGBoost
   - Logistic Regression
   - LightGBM
4. Select and promote the best model using validation business performance under the current policy framing.
5. Add interpretability artifacts:
   - SHAP summary plot for the promoted model if supported
   - fallback feature importance output if SHAP is not feasible
6. Add a Streamlit dashboard that loads saved artifacts without retraining.
7. Add `requirements.txt` and targeted tests for split logic, leakage sanity, schema, and metrics.
8. Rerun the end-to-end pipeline, regenerate reports, update README, and write a final upgrade summary.

## Repo Constraints / Missing Pieces

- The repository does not currently include CI configuration or an environment lockfile; I will add minimal reproducibility support rather than a full packaging overhaul.
- There are no existing plotting utilities, so lightweight plotting will be added.
- The current scoring pipeline assumes a single promoted model. I will preserve that pattern and update promotion to point to the best validated model.
- Backtesting will reuse the canonical feature table because features and labels are already computed point-in-time for each row; recomputing from raw data for every fold would add runtime without changing the row-level leakage guarantees.
