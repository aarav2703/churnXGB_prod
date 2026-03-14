# ChurnXGB - Leakage-Aware Churn Targeting Under Budget Constraints

I built this project to answer a version of churn modeling that feels much closer to how a retention team would actually use a model in practice.

Instead of asking only, "Can I predict who will churn?", I framed the problem as, "If a team can only contact 5% to 20% of customers, which customers should they prioritize to protect the most value at risk?"

That framing shaped most of my design choices:
- I built point-in-time customer-month snapshots so features only use information available at decision time.
- I defined churn operationally as no purchase in the next 90 days after the cutoff timestamp.
- I evaluated models not just with standard ML metrics, but also with Value-at-Risk@K and related top-K targeting metrics.
- I compared a more complex model family against simpler baselines and heuristic policies instead of assuming the most complex model would win.
- I added MLflow logging, drift outputs, and a lightweight Streamlit dashboard so the repo feels like an end-to-end applied ML project rather than only a modeling notebook.

The repository now includes:
- point-in-time feature generation and 90-day churn labeling
- multi-model training across XGBoost, Logistic Regression, and LightGBM
- business-aware evaluation with Value-at-Risk@K plus broader ML metrics
- rolling temporal backtesting
- MLflow experiment logging and model promotion
- interpretability artifacts
- PSI-based drift monitoring
- a Streamlit dashboard that loads saved artifacts

## Why I Framed The Problem This Way

Most churn projects stop at probability ranking, but I wanted this project to reflect the real decision constraint a retention team faces: limited outreach capacity.

If a team can only intervene on a small share of customers, then a model with a decent AUC is not automatically useful. What matters is whether the ranking captures customers who are both likely to churn and important enough to prioritize.

That is why I used the policy:

`policy_ml = P(churn) * value_pos`

where `value_pos` is a pre-decision value proxy based on trailing 90-day revenue clipped at zero.

I made that choice deliberately because I did not want to "cheat" by using future value information that would not be available at scoring time.

## Problem Framing

Each row is a customer-month snapshot. Let `T` be the customer's last purchase timestamp in that month.

- Features use data available up to `T`
- Labels use events strictly after `T`
- Churn is defined as no purchase in the next 90 days after `T`

I used this setup because it mirrors how a churn model would actually be deployed and helps avoid obvious leakage from future behavior.

## Dataset

- Source: Online Retail II transactional dataset
- Raw grain: transaction lines
- Current processed modeling table: 26,993 customer-month rows
- Current event table: 44,571 customer events
- Observed monthly span in processed data: `2009-12` through `2011-12`

I started from raw transaction lines and converted them into a customer-month prediction problem because that gave me a realistic decision grain for retention targeting while still preserving temporal ordering.

## Feature Engineering

I computed features at the customer-month cutoff using only prior customer behavior:

- trailing revenue sums: `rev_sum_30d`, `rev_sum_90d`, `rev_sum_180d`
- purchase frequency: `freq_30d`, `freq_90d`
- revenue volatility: `rev_std_90d`
- return behavior: `return_count_90d`
- average order value proxy: `aov_90d`
- recency gap: `gap_days_prev`

I focused on a compact, interpretable feature set rather than trying to build a very wide table with weakly justified variables. The goal was to show solid point-in-time feature engineering and keep the project understandable in an interview.

## Models I Compared

I did not want the project to rely on a single model family, so I compared:

- `xgboost`
- `logistic_regression`
- `lightgbm`

I also kept non-ML policy baselines because I wanted the learned models to earn their complexity:

- recency-based targeting
- RFM-style targeting
- random targeting baseline

Including logistic regression was especially important to me because it gives a simpler benchmark and helps answer whether the tree models are actually adding value or just adding complexity.

## Evaluation

I kept the original business-first framing and then expanded the evaluation suite so the project would read more strongly to both data science recruiters and applied ML reviewers.

### Business / targeting metrics

- Value-at-Risk@K
- fraction of total value-at-risk captured
- Precision@K
- Recall@K
- Lift@K

### ML metrics

- ROC-AUC
- PR-AUC
- Brier score
- calibration curve data

### Saved plots

Generated in `reports/figures/`:
- ROC curve
- Precision-Recall curve
- lift curve
- calibration curve
- feature importance plot

I added the broader metric suite because I wanted the project to show both business awareness and ML rigor. VaR@K is the metric I would emphasize in a retention setting, but PR-AUC, Brier score, and calibration still matter if I want to show that the underlying probabilities are sensible.

## Current Results

### Holdout comparison at 10% budget

| model | val_value_at_risk | test_value_at_risk | test_roc_auc | test_pr_auc | test_brier_score |
|:--|--:|--:|--:|--:|--:|
| logistic_regression | 97,853.45 | 165,919.06 | 0.7335 | 0.5819 | 0.2016 |
| xgboost | 76,760.36 | 164,787.80 | 0.7233 | 0.5805 | 0.2077 |
| lightgbm | 72,032.55 | 164,581.86 | 0.7205 | 0.5749 | 0.2089 |

Promoted model: `logistic_regression`

One result I found interesting is that logistic regression won the current comparison. I like that outcome for this portfolio project because it shows I was willing to let the validation metric choose the best model instead of forcing the more complex model to be the headline result.

### Promoted model test targeting metrics

| budget_k | value_at_risk | var_covered_frac | precision_at_k | recall_at_k | lift_at_k |
|:--|--:|--:|--:|--:|--:|
| 5% | 108,496.47 | 0.1738 | 0.4604 | 0.0607 | 1.2140 |
| 10% | 165,919.06 | 0.2658 | 0.4065 | 0.1072 | 1.0717 |
| 20% | 260,599.82 | 0.4175 | 0.3939 | 0.2078 | 1.0385 |

At the 10% budget level, the promoted model captures `165,919.06` in value at risk on the holdout split. Compared with the best heuristic baseline at the same budget, that is roughly a `54.8%` uplift in captured value at risk.

### Temporal backtesting

I added rolling expanding-window backtesting to move beyond a single holdout split and check whether performance is reasonably stable across time.

Backtesting was run across 9 chronological folds:

- `2010-06_2010-07`
- `2010-08_2010-09`
- `2010-10_2010-11`
- `2010-12_2011-01`
- `2011-02_2011-03`
- `2011-04_2011-05`
- `2011-06_2011-07`
- `2011-08_2011-09`
- `2011-10_2011-11`

Backtest outputs are written to:
- `reports/backtest_detail.csv`
- `reports/backtest_summary.csv`
- `reports/backtest_summary.md`

I added this because I wanted a stronger answer to the question, "Does this model still work when the time window shifts?" That felt like a more credible applied ML signal than relying only on one train/validation/test split.

## Interpretability And Monitoring

I wanted the repo to show more than training and scoring, so I added:

- feature importance artifacts in `reports/feature_importance.csv`
- feature analysis writeup in `reports/feature_analysis.md`
- drift monitoring outputs in `reports/monitoring/`

For this version of the project, interpretability is lightweight by design. The goal was to make model behavior easier to inspect without turning the repo into a full interpretability platform.

## Dashboard

I added a Streamlit dashboard in `dashboard/app.py` so the saved outputs can be explored without retraining the models.

The dashboard includes:
- executive summary
- policy simulator
- model performance charts
- explainability view
- customer risk explorer
- drift monitoring

I made the dashboard artifact-driven on purpose. For a recruiter or reviewer, that is a much smoother demo flow than requiring a live retrain just to inspect results.

### Dashboard Preview

The first dashboard view highlights the executive summary, model comparison table, and budget-based policy simulator. I like this screen as a project overview because it quickly communicates the promoted model, the top-line metrics, and how the targeting policy behaves at a chosen budget.

![Dashboard overview showing executive summary, model comparison, and policy simulator](reports/figures/dashboard_overview.png)

The second dashboard view focuses on explainability and customer-level exploration. It shows the promoted model's feature importance, the ranked customer risk table, and the target flags used for downstream action lists.

![Dashboard explainability and customer risk explorer view](reports/figures/dashboard_explain_explore.png)

## Repository Structure

```text
config/
  config.yaml

data/
  raw/
  interim/
  processed/

dashboard/
  app.py

models/
  registry/
  promoted/

outputs/
  predictions/
  targets/

reports/
  evaluation/
  figures/
  monitoring/
  *.md / *.csv summaries

src/churnxgb/
  data/
  features/
  labeling/
  baselines/
  modeling/
  policy/
  evaluation/
  monitoring/
  pipeline/
  split/
  utils/

tests/
```

## How To Run

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the pipeline

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m churnxgb.pipeline.build_features
python -m churnxgb.pipeline.train
python -m churnxgb.pipeline.score
```

Or use the Windows runbook:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

### 3. Run tests

```powershell
pytest -q
```

### 4. Launch the dashboard

```powershell
streamlit run dashboard/app.py
```

The dashboard reads saved artifacts and does not retrain models live.

## Key Outputs

### Training and evaluation

- `reports/model_comparison.csv`
- `reports/model_comparison.md`
- `reports/model_eval_summary.md`
- `reports/backtest_summary.csv`
- `reports/backtest_summary.md`
- `reports/feature_importance.csv`
- `reports/feature_analysis.md`
- `reports/final_upgrade_summary.md`
- `reports/final_results_summary.md`
- `reports/evaluation/classification_metrics.csv`
- `reports/evaluation/policy_metrics_all_models.csv`

### Monitoring and scoring

- `reports/monitoring/reference_profile.json`
- `reports/monitoring/drift_latest.json`
- `outputs/predictions/predictions_all.parquet`
- `outputs/targets/targets_all_k05.parquet`
- `outputs/targets/targets_all_k10.parquet`
- `outputs/targets/targets_all_k20.parquet`

### Promotion and experiment tracking

- `models/promoted/production.json`
- `mlflow.db`
- `mlruns/`

## Reproducibility Notes

- The pipeline logs a SHA-256 hash of the processed feature table as `data_version`.
- Training logs metrics and artifacts to MLflow.
- The promoted model record points to the selected MLflow run id.
- `requirements.txt` captures the runtime dependencies used for the upgraded pipeline.

I also added tests for temporal splitting, leakage-safe feature selection, feature schema, and metric sanity because I wanted a few targeted safeguards without turning this into a heavy software engineering project.

## Known Limitations

- Delayed-label performance tracking after future labels mature is not yet implemented.
- Monitoring is intentionally lightweight: feature PSI plus score distribution summary stats.
- The dashboard is designed for portfolio/demo use, not production serving.
- Environment compatibility could still be tightened further, especially around package version consistency.

## What I Would Improve Next

If I continued this project, the next improvements I would prioritize are:

- lightweight hyperparameter search using the temporal validation framework
- explicit probability calibration comparisons
- cohort-level error analysis
- delayed-label evaluation of scored cohorts
- slightly stronger dashboard storytelling and drift alert summaries

I think those would add the most value for data science and applied ML recruiting without overcomplicating the repository.
