# ChurnXGB: Leakage-Aware Churn Targeting Under Budget Constraints

This project started as a churn modeling exercise, but I ended up treating it more like a small decision system.

The question I cared about was not just "who is likely to churn?" It was:

If a retention team can only contact 5% to 20% of customers, who should they prioritize to protect the most value?

That changed the shape of the whole project. Instead of building one model and stopping at AUC, I built a point-in-time customer-month dataset, compared multiple model families, evaluated them with top-K targeting metrics, and then extended the repo into something that can score, explain, simulate policies, monitor drift, and expose all of that through an API and frontend.

## What This Repo Does

At a high level, the repo now supports:

- offline feature building, training, and scoring
- model comparison across `xgboost`, `logistic_regression`, and `lightgbm`
- budget-aware targeting using both `policy_ml` and `policy_net_benefit`
- per-customer explanation through the API
- drift monitoring with latest snapshot, alert summaries, and drift history
- a FastAPI backend
- a Streamlit dashboard
- a React decision UI
- an optional LLM query layer that sits on top of the backend tools

The live runtime artifacts are written to `.runtime/`.

Older root-level `models/`, `outputs/`, and `reports/` paths may still exist in the repo from earlier iterations. I kept them for continuity, but the current pipeline uses `.runtime/...` as the source of truth.

## How I Framed the Problem

Each row is a customer-month snapshot.

Let `T` be the customer's last purchase timestamp in that month.

- features use information available up to `T`
- labels use events strictly after `T`
- churn is defined as no purchase in the next 90 days after `T`

I used that setup because I wanted to avoid the usual leakage problems that show up when churn is framed too loosely. It also felt closer to how a real retention team would use a model: make a decision at a cutoff date, not after seeing the future.

## Dataset

- Source: Online Retail II
- Raw grain: transaction lines
- Current processed modeling table: 26,993 customer-month rows
- Current customer-event table: 44,571 rows
- Observed monthly span: `2009-12` through `2011-12`

I turned the raw transactions into a customer-month prediction problem because it gave me a practical decision grain for targeting while preserving the time ordering.

## Features

I kept the feature set fairly compact and interpretable:

- trailing revenue: `rev_sum_30d`, `rev_sum_90d`, `rev_sum_180d`
- frequency: `freq_30d`, `freq_90d`
- revenue volatility: `rev_std_90d`
- returns: `return_count_90d`
- average order value proxy: `aov_90d`
- recency gap: `gap_days_prev`

I deliberately did not turn this into a huge feature factory. The goal here was a solid, leakage-aware baseline that I could still explain clearly.

## Models and Policies

The repo compares:

- `xgboost`
- `logistic_regression`
- `lightgbm`

I also kept heuristic baselines:

- recency targeting
- RFM-style targeting
- random targeting

That mattered to me because I wanted the learned models to earn their complexity.

The original targeting score was:

`policy_ml = P(churn) * value_pos`

where `value_pos` is a trailing-value proxy based on pre-decision revenue.

The current repo still computes `policy_ml`, but it also computes a cost-aware decision score:

`policy_net_benefit = expected_retained_value - expected_cost`

using explicit config-driven economics and customer-level heterogeneity from existing recency, frequency, and value features.

## What the Pipeline Looks Like

The main pipeline is:

1. `python -m churnxgb.pipeline.build_features`
2. `python -m churnxgb.pipeline.train`
3. `python -m churnxgb.pipeline.score`

That pipeline:

- builds the point-in-time feature table
- trains and compares models
- logs runs to MLflow
- promotes the selected model
- scores the saved feature table
- writes predictions, target lists, and drift outputs

The current runtime writes to:

- `.runtime/data/...`
- `.runtime/models/...`
- `.runtime/outputs/...`
- `.runtime/reports/...`

## Current Runtime State

From the latest verified runtime artifacts:

- promoted model: `xgboost`
- selected registry name: `churn_xgboost_v1`
- chosen budget: `10%`
- selection policy: `policy_net_benefit`

Those values come from:

- `.runtime/reports/training_manifest.json`
- `.runtime/models/promoted/production.json`

## Results

These values come from a recent verified run in `.runtime/`, so they should be treated as current runtime outputs rather than timeless fixed benchmarks.

### Holdout comparison at 10% budget

| model | val_value_at_risk | test_value_at_risk | test_roc_auc | test_pr_auc | test_brier_score |
|:--|--:|--:|--:|--:|--:|
| xgboost | 75,325.34 | 165,488.88 | 0.7233 | 0.5805 | 0.2077 |
| lightgbm | 73,281.12 | 162,022.38 | 0.7205 | 0.5749 | 0.2089 |
| logistic_regression | 98,494.27 | 159,082.89 | 0.7335 | 0.5819 | 0.2016 |

One thing I actually like about this result is that the "best" model depends on what I care about.

- logistic regression is strongest on some conventional metrics
- xgboost is the currently promoted model because the runtime selection is aligned with the deployed decision layer

That feels more honest than forcing one model to win everything.

### Promoted model targeting metrics on test

| budget_k | value_at_risk | var_covered_frac | precision_at_k | recall_at_k | lift_at_k |
|:--|--:|--:|--:|--:|--:|
| 5% | 109,553.17 | 0.1755 | 0.4737 | 0.0624 | 1.2491 |
| 10% | 165,488.88 | 0.2651 | 0.4281 | 0.1129 | 1.1286 |
| 20% | 261,772.74 | 0.4194 | 0.4054 | 0.2138 | 1.0691 |

At 10% budget, the promoted runtime model captures about `165.5k` in value at risk on the test split.

The repo also now tracks `net_benefit_at_k`, which matters because I did not want the decision layer to stop at VaR alone.

### Decision-layer behavior

The newer decision layer is not just a relabeling of the same ranking. In the latest runtime outputs, the top-K overlap between `policy_ml` and `policy_net_benefit` is below 100%, so the cost-aware ranking does change who gets targeted:

- 5% budget overlap: `86.6%`
- 10% budget overlap: `89.4%`
- 20% budget overlap: `90.6%`

That was an important upgrade for me because I did not want the "decision layer" to be purely cosmetic.

### Backtesting

I added rolling expanding-window backtesting because I did not want the project to depend on a single temporal split.

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

Outputs:

- `.runtime/reports/backtest_detail.csv`
- `.runtime/reports/backtest_summary.csv`
- `.runtime/reports/backtest_summary.md`

## API and App Layer

The backend is in FastAPI and sits on top of the offline artifacts plus the live scoring path.

Current endpoints include:

- `GET /health`
- `GET /model-summary`
- `GET /model-comparison`
- `GET /policy-metrics`
- `GET /feature-importance`
- `GET /targets/{budget_pct}`
- `GET /drift/latest`
- `GET /drift/history`
- `GET /predictions`
- `GET /customers/explain`
- `POST /predict`
- `POST /explain`
- `POST /simulate-policy`
- `POST /simulate-experiment`
- `POST /llm/query`

Important distinction:

- `/predict` and `/explain` use the backend scoring/explanation logic directly
- many of the summary and simulation endpoints are artifact-backed, so they depend on the offline pipeline having been run first

Also important:

- `/predict` expects engineered feature columns that match the inference contract
- it does **not** accept raw transactions

## Interpretability and Monitoring

I did not want this repo to be only "train once, report AUC, stop."

So I added:

- feature importance outputs
- per-customer explanation through the API
- drift monitoring

Monitoring is still lightweight, but it now includes:

- latest drift snapshot
- alert summaries
- persisted drift history

Key files:

- `.runtime/reports/feature_importance.csv`
- `.runtime/reports/monitoring/reference_profile.json`
- `.runtime/reports/monitoring/drift_latest.json`
- `.runtime/reports/monitoring/drift_history.csv`

For the logistic-regression explanation path, contributions are interpreted relative to the standardized baseline, which corresponds to the training-data mean after scaling.

## Frontends

There are two UI layers in the repo.

### Streamlit

The Streamlit app in `dashboard/app.py` is still useful as a quick artifact-driven demo:

- executive summary
- policy simulator
- model performance
- explainability
- customer risk exploration
- drift monitoring

Current screenshot files already in the repo:

![Streamlit dashboard overview](reports/figures/dashboard_overview.png)
![Streamlit explainability and customer explorer](reports/figures/dashboard_explain_explore.png)

### React frontend

The React app in `frontend/` is the more customized interface.

It now includes:

- overview
- targeting simulator
- customer explanation
- drift monitoring
- experiment simulation
- Chat / Ask

The React UI is still backend-driven. It does not compute scores in the browser.

The goal of the React layer is to make the repo feel more like a decision workflow instead of a set of raw tables. So it includes:

- decision snapshot cards
- bar-style comparison visuals
- overlap and treatment-mix visuals
- clearer customer-explanation states
- a chat panel on top of backend tools

The customer explanation page depends on saved scored predictions. If the offline scoring pipeline has not been run, the page will show an explanatory empty/error state rather than inventing data.

If I include screenshots for the React app in the repo, the two most useful ones are:

- the overview page, which shows the decision snapshot cards and policy strength visuals
- the Chat / Ask page, which shows the tool-routed LLM layer and raw tool outputs

I intentionally keep both frontends in the README because they serve slightly different purposes:

- Streamlit is the fastest artifact-driven review layer
- React is the more customized API-driven decision UI

Recommended screenshot paths for the React app:

- `docs/screenshots/react_overview.png`
- `docs/screenshots/react_chat.png`

I have not replaced the Streamlit section with React because I still think the Streamlit app is useful as a lightweight demo of saved outputs, while React is better for showing the more polished decision workflow.

## LLM Layer

I added a lightweight LLM layer on top of the backend, but I was careful not to let it replace the ML system.

The design is:

User -> LLM query endpoint -> backend tool routing -> existing API/data -> summary

That means:

- the LLM does not generate `churn_prob`
- the LLM does not generate `policy_ml`
- the LLM does not generate `policy_net_benefit`
- it only summarizes grounded backend outputs

The tool-aware layer lives in:

- `src/churnxgb/llm/`

and the public endpoint is:

- `POST /llm/query`

If `DEEPSEEK_API_KEY` is set, the backend can call DeepSeek for the final summary. If not, it falls back to a deterministic grounded summary from the same tool outputs.

## How To Run

### 1. Install dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the offline pipeline

```powershell
$env:PYTHONPATH = "$PWD\src"

python -m churnxgb.pipeline.build_features
python -m churnxgb.pipeline.train
python -m churnxgb.pipeline.score
```

Or use:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

The current pipeline writes live runtime outputs to `.runtime/`.

### 3. Run tests

```powershell
pytest -q
```

### 4. Launch the API

```powershell
$env:PYTHONPATH = "$PWD\src"
uvicorn churnxgb.api.app:app --host 0.0.0.0 --port 8000
```

Example:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

### 5. Launch Streamlit

```powershell
streamlit run dashboard/app.py
```

### 6. Launch the React frontend

```powershell
cd frontend
cmd /c npm.cmd install
cmd /c npm.cmd run dev
```

To build it:

```powershell
cmd /c npm.cmd run build
```

### 7. Optional: enable the LLM query layer

Before starting the API, set:

```powershell
$env:DEEPSEEK_API_KEY="your_key_here"
```

If you do not set it, the `POST /llm/query` flow still works, but the final answer uses the deterministic fallback summary instead of an external LLM call.

### 8. Docker

There is still a Dockerized API entrypoint for local serving:

```powershell
docker build -t churnxgb-api .
docker run --rm -p 8000:8000 churnxgb-api
```

I still think of Docker here as a local convenience, not a full deployment story.

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

frontend/
  src/
  dist/  # created after frontend build

.runtime/
  data/
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
    *.json / *.csv / *.md summaries

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
  llm/

tests/
```

## Key Outputs

### Runtime outputs

- `.runtime/reports/model_comparison.csv`
- `.runtime/reports/training_manifest.json`
- `.runtime/reports/evaluation/classification_metrics.csv`
- `.runtime/reports/evaluation/policy_metrics_all_models.csv`
- `.runtime/reports/feature_importance.csv`
- `.runtime/reports/monitoring/reference_profile.json`
- `.runtime/reports/monitoring/drift_latest.json`
- `.runtime/reports/monitoring/drift_history.csv`
- `.runtime/outputs/predictions/predictions_all.parquet`
- `.runtime/outputs/predictions/predictions_inference.parquet`
- `.runtime/outputs/targets/targets_all_k05.parquet`
- `.runtime/outputs/targets/targets_all_k10.parquet`
- `.runtime/outputs/targets/targets_all_k20.parquet`
- `.runtime/models/promoted/production.json`

### Tracking

- `mlruns_store/`

### Legacy / deprecated locations

These may still exist from earlier iterations, but they are no longer the primary live runtime outputs:

- `models/promoted/production.json`
- `models/registry/...`
- `outputs/...`
- `reports/...`
- `mlflow.db`
- `mlruns/`

## Reproducibility Notes

- the pipeline logs a SHA-256 hash of the processed feature table as `data_version`
- training logs metrics and artifacts to MLflow file-store under `mlruns_store/`
- the promoted model record points to the selected run id and local promoted registry path under `.runtime/models/promoted/production.json`
- each saved model carries an `inference_contract.json` file defining required inference inputs and clean prediction outputs

## Known Limitations

- delayed-label performance tracking after future labels mature is not implemented yet
- monitoring is still lightweight compared with a real production monitoring stack
- the FastAPI service is still minimal: no authentication, request logging, or deployment infrastructure
- the Streamlit and React frontends are meant for portfolio/demo use, not production serving
- the experiment simulation layer is assumption-driven business simulation, not causal inference from observed treatment/control data
- the LLM layer is only as good as the backend tool outputs and available artifacts

## What I Would Improve Next

If I kept working on this, I would probably focus on:

- better probability calibration analysis
- delayed-label evaluation of scored cohorts
- more structured cohort/error analysis
- cleaner deployment setup around the API
- a stronger frontend narrative layer, especially around recommendations and analyst workflow

Those feel like the highest-value next steps without turning the project into something much bigger than it needs to be.
