# ChurnXGB

ChurnXGB is a churn-targeting project built around a simple business question:

If a company can only contact a limited number of customers, who should it target to protect the most value?

That is why this repo is not just a churn classifier. It is an offline decision system built on customer-month snapshots, temporal validation, calibrated probabilities, budget-based targeting, and a small product layer on top through FastAPI and React.

## What This Project Does

The system:

- builds point-in-time customer-month features from transaction data
- predicts 90-day churn
- compares `xgboost`, `lightgbm`, and `logistic_regression`
- calibrates probabilities before deployment
- ranks customers with both `policy_ml` and `policy_net_benefit`
- evaluates both model quality and targeting quality
- runs backtests and segment-level analysis
- tracks feature drift and decision drift
- serves saved outputs through FastAPI
- exposes the results in Streamlit and React dashboards

The main point of the project is that churn prediction by itself is not enough. If the model says 5,000 customers are risky but the business can only contact 300 of them, then ranking and decision quality matter just as much as the classifier.

## Why I Built It This Way

I kept the project centered on customer-month snapshots because that is a practical unit for retention analysis. It is detailed enough to capture behavior over time, but still simple enough to train, evaluate, and explain cleanly.

I also wanted the project to reflect how an applied data science system usually works in practice:

- data is built in an offline pipeline
- time leakage has to be controlled carefully
- model quality is not the same thing as business usefulness
- outputs should be saved and inspectable
- a model is more convincing when it can be served, monitored, and explained

So the repo is intentionally broader than a single notebook, but still small enough to understand end to end.

## Problem Setup

Each row is a customer-month snapshot.

For each row:

- features use only information available up to that row's month-end reference point
- labels use behavior strictly after that point
- churn means no purchase in the next 90 days

This is important because it keeps the setup point-in-time correct. A churn project can look very strong on paper if leakage sneaks in through features or splits, so the pipeline is built to avoid that.

The project keeps the data grain fixed at customer-month. I did not switch it to online event scoring or a different architecture.

## Repository Structure

The repo follows a simple production-style shape:

1. Offline pipeline
   - build features
   - train and compare models
   - calibrate probabilities
   - score the saved feature table
   - write reports and saved artifacts

2. FastAPI layer
   - prediction and explanation endpoints
   - budget frontier, segment, backtest, and drift endpoints
   - policy simulation endpoints over saved outputs

3. Frontends
   - Streamlit for artifact review
   - React for a more interactive decision dashboard

I kept this structure because it makes the project feel closer to a small applied system rather than a one-off experiment.

## Data And Features

Dataset:

- source: Online Retail II
- raw grain: transaction lines
- modeling grain: customer-month
- latest verified processed table size: `26,993` rows

Core feature families:

- revenue windows: `rev_sum_30d`, `rev_sum_90d`, `rev_sum_180d`
- frequency windows: `freq_30d`, `freq_90d`
- volatility: `rev_std_90d`
- returns: `return_count_90d`
- average order value proxy: `aov_90d`
- recency: `gap_days_prev`

I kept the feature set fairly compact on purpose. I wanted enough signal to make the system useful, but not so many features that it became hard to reason about what the model was doing.

## Models And Decision Scores

The training pipeline compares:

- `xgboost`
- `lightgbm`
- `logistic_regression`

It also keeps simple heuristic baselines for comparison.

The two main decision scores are:

- `policy_ml = churn_prob * value_pos`
- `policy_net_benefit = expected_retained_value - expected_cost`

I kept both because they answer slightly different questions.

`policy_ml` is the cleaner "risk times value" score.

`policy_net_benefit` is the more business-facing score because it tries to account for intervention cost and expected retained value.

That is also why the project uses budgeted decision metrics in addition to standard classification metrics.

## Why Calibration Was Added

One of the biggest upgrades was probability calibration.

I added it because the project does not use predicted probabilities only for reporting. It uses them inside downstream targeting logic. That means poorly calibrated probabilities can hurt the decision layer even when ranking still looks decent.

What changed:

- the training pipeline fits a calibrator after the base model
- the promoted model artifact stores calibration information
- scoring uses calibrated probabilities by default
- evaluation keeps both raw and calibrated comparisons

From the latest verified run:

- promoted model: `lightgbm`
- promoted artifact: `churn_lightgbm_v1`
- calibration method: `platt`
- chosen budget: `10%`
- selection policy: `policy_net_benefit`

Calibration improved the decision layer in the latest run:

- validation net benefit improved from about `14.2k` raw to `37.3k`
- test net benefit improved from about `21.7k` raw to `53.1k`
- validation Brier score improved from `0.1832` to `0.1818`
- test Brier score improved from `0.2170` to `0.2077`

That does not mean calibration improves every metric in every case. The reason I kept it is that in this project it improved probability quality and the final targeting setup.

## Why Segment Analysis Was Added

I added segment-level evaluation because average metrics can hide where the model is and is not actually useful.

The project writes segment outputs using existing behavioral features:

- value bands from `rev_sum_90d`
- recency buckets from `gap_days_prev`
- frequency buckets from `freq_90d`

Saved output:

- `.runtime/reports/evaluation_segments.csv`

Latest verified run:

- `27` segment rows

Why I think this matters:

- high-value segments carry most of the economic upside
- low-value segments can still show lift but weak economics
- this helps separate model accuracy from actionability

That is much closer to how a business team would actually ask follow-up questions after a first model run.

## Why Backtesting Was Added

I added expanding-window backtesting because a single validation split can be misleading, especially in temporal problems.

Saved outputs:

- `.runtime/reports/backtest_detail.csv`
- `.runtime/reports/backtest_summary.csv`

Backtesting matters here because it shows whether the model and the policy stay reasonably stable across different time windows. That makes the results more believable than one lucky holdout.

## Why Drift Monitoring Was Added

I added two monitoring layers:

1. Feature drift
   - PSI by feature
   - drift summary and alert counts

2. Decision drift
   - selected share by month
   - average churn score in top-K
   - average value in top-K
   - monthly VaR@K trend

Saved outputs:

- `.runtime/reports/monitoring/drift_latest.json`
- `.runtime/reports/monitoring/drift_history.csv`
- `.runtime/reports/decision_drift.csv`

I added decision drift specifically because I did not want monitoring to stop at "feature distribution changed." In a targeting project, it is also useful to see whether the selected population and its economic profile are changing over time.

From the latest verified run:

- feature drift status was `ok`
- latest top PSI was `0.082`
- decision drift had `75` monthly budget rows

## Latest Verified Results

These numbers come from a full pipeline run completed in this repo.

### Best model at the selected budget

- model: `lightgbm`
- run id: `45ff38b82e0e433fbc18d065311e4b9a`
- budget: `10%`
- selection policy: `policy_net_benefit`

### Holdout comparison at 10% budget

| model | val_net_benefit_at_k | test_net_benefit_at_k | test_value_at_risk | test_roc_auc | test_brier_score |
|:--|--:|--:|--:|--:|--:|
| lightgbm | 37,265.96 | 53,074.73 | 128,102.77 | 0.7167 | 0.2077 |
| logistic_regression | 32,486.32 | 38,294.51 | 152,092.72 | 0.7338 | 0.2016 |
| xgboost | 34,896.93 | 50,224.41 | 143,190.49 | 0.7224 | 0.2064 |

There are a couple of useful takeaways here:

- logistic regression stays competitive on standard classification metrics
- lightgbm wins on the chosen decision objective in the verified run
- this is exactly why I did not want to judge the system only by ROC-AUC

### Budget frontier for the promoted model

| budget_k | value_at_risk | net_benefit_at_k | targeted_count | precision_at_k | recall_at_k | lift_at_k |
|:--|--:|--:|--:|--:|--:|--:|
| 5% | 77,688.30 | 41,899.80 | 139 | 0.2014 | 0.0266 | 0.5311 |
| 10% | 128,102.77 | 53,074.73 | 278 | 0.2338 | 0.0617 | 0.6165 |
| 20% | 235,673.05 | 67,672.40 | 556 | 0.2824 | 0.1490 | 0.7445 |

This frontier is useful because it makes the budget tradeoff concrete. The business can see that going from 10% to 20% increases captured value and net benefit, but it also doubles the targeting volume.

## API

The FastAPI backend serves both live scoring and saved decision artifacts.

Main endpoints:

- `GET /health`
- `GET /model-summary`
- `GET /model-comparison`
- `GET /policy-metrics`
- `GET /frontier`
- `GET /segments`
- `GET /backtest`
- `GET /targets/{budget_pct}`
- `GET /predictions`
- `GET /customers/explain`
- `GET /drift/latest`
- `GET /drift/history`
- `GET /drift/decision`
- `POST /predict`
- `POST /explain`
- `POST /simulate-policy`
- `POST /simulate-experiment`
- `POST /llm/query`
- `POST /llm/explain_customer`

I kept the helper ask/explain endpoints in the repo, but they sit on top of existing backend outputs. They are not the core of the project, and they do not replace the actual model or policy logic.

## Dashboard Screenshots

### Executive Summary

![Executive Summary](./docs/screenshots/executive-summary.png)

### Targeting Strategy

![Targeting Strategy](./docs/screenshots/targeting-strategy.png)

### Model Performance

![Model Performance](./docs/screenshots/model-performance.png)

### Segment Analysis

![Segment Analysis](./docs/screenshots/segment-analysis.png)

### Decision Card And Model Comparison

![Decision Card And Model Comparison](./docs/screenshots/decision-card-model-comparison.png)

### Ask / Explain

![Ask / Explain](./docs/screenshots/ask-explain.png)

## How To Run

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Set `PYTHONPATH`

```powershell
$env:PYTHONPATH = "$PWD\src"
```

### 3. Build features

```powershell
python -m churnxgb.pipeline.build_features
```

Builds the customer-month feature table from the raw retail data.

### 4. Train models

```powershell
python -m churnxgb.pipeline.train
```

Trains candidate models, calibrates probabilities, writes reports, and promotes the selected model.

### 5. Score the saved feature table

```powershell
python -m churnxgb.pipeline.score
```

Scores the saved table, writes predictions and target lists, and updates drift outputs.

### 6. Run tests

```powershell
pytest -q
```

### 7. Start the API

```powershell
uvicorn churnxgb.api.app:app --host 0.0.0.0 --port 8000
```

### 8. Start Streamlit

```powershell
streamlit run dashboard/app.py
```

### 9. Start the React frontend

```powershell
cd frontend
cmd /c npm.cmd install
cmd /c npm.cmd run dev
```

Optional build check:

```powershell
cd frontend
cmd /c npm.cmd run build
```

## Key Runtime Outputs

Model and evaluation:

- `.runtime/reports/model_comparison.csv`
- `.runtime/reports/training_manifest.json`
- `.runtime/reports/calibration_summary.md`
- `.runtime/reports/evaluation_segments.csv`
- `.runtime/reports/evaluation/lightgbm_test_frontier.csv`
- `.runtime/reports/backtest_detail.csv`
- `.runtime/reports/backtest_summary.csv`

Monitoring:

- `.runtime/reports/monitoring/drift_latest.json`
- `.runtime/reports/monitoring/drift_history.csv`
- `.runtime/reports/decision_drift.csv`

Scoring outputs:

- `.runtime/outputs/predictions/predictions_all.parquet`
- `.runtime/outputs/targets/targets_all_k05.parquet`
- `.runtime/outputs/targets/targets_all_k10.parquet`
- `.runtime/outputs/targets/targets_all_k20.parquet`

## Limitations

- policy simulation is assumption-driven, not causal inference
- experiment simulation is also assumption-driven
- delayed-label production monitoring is not implemented
- the API is local/dev oriented, not deployment infrastructure
- segment logic is intentionally simple

## Verdict

I think this is a strong grad-student data science project.

The results are good, but they are not suspiciously perfect:

- ROC-AUC is around `0.72` to `0.73`
- calibration improves Brier score and decision value
- the budget frontier shows meaningful tradeoffs
- segment analysis shows where the economics are strong and weak
- backtests and drift outputs make the system more believable

So my honest conclusion is:

- for a data science graduate student, yes, this is good enough
- for internships and entry-level DS roles, it is definitely strong enough
- for more senior applied ML roles, it would not be enough on its own, but it is a good foundation

The strongest part is not one metric. The strongest part is that the project shows the whole chain: point-in-time data setup, leakage control, model comparison, calibration, decision metrics, monitoring, API serving, and a working dashboard.
