# ChurnXGB — Customer Churn Risk Scoring Under Budget Constraints (Productionized)

The core idea is simple: churn modeling is a decision problem. If you can only contact 5–20% of customers, accuracy alone doesn’t matter—what matters is how much customer value you can proactively protect with a limited outreach budget.

This repo contains an end-to-end system that:
* builds point-in-time customer-month snapshots (no leakage),
* trains an XGBoost churn probability model,
* ranks customers using a Value-at-Risk policy under budget constraints,
* logs experiments + artifacts to MLflow (with dataset hashing),
* “promotes” a run to production and scores from the promoted MLflow run, and
* writes a drift report each scoring run (feature PSI + score distribution stats).

## What problem am I solving?
Most churn projects stop at “predict churn probability.” In practice, retention teams ask:

> “If I can only reach out to K% of customers, who should I contact to maximize retained revenue?”

So I frame the objective as target selection under a fixed budget, not pure classification.

## Dataset
* **Source:** Online Retail II transactional dataset
* **Raw grain:** transaction lines

I transform it into:
* invoice-level records
* deduplicated customer “events”
* customer-month snapshots

*The raw Online Retail II CSV is placed in `data/raw/` and is ignored via `.gitignore` to prevent committing large datasets. All intermediate artifacts are reproducible from this source.*

## Operational churn definition (no explicit churn label)
There is no explicit churn label, so I define it operationally:

* Each row is a customer-month snapshot.
* Let `T` be the customer’s last purchase timestamp in that month.
* A customer is labeled churned at time `T` if they make no purchase in the next 90 days after `T`.

**Key constraint I enforced throughout:**
* All features are computed using data available **up to** `T`
* Labels are computed strictly using events **after** `T`

This mirrors how churn models are deployed in practice and prevents leakage.

## Why the project uses “Value at Risk @ K%” instead of just AUC
I evaluate models with Value at Risk @ K% (VaR@K):

1. rank customers by a targeting / intervention policy
2. take the top K% (K = 5%, 10%, 20%)
3. compute how much positive customer value at risk exists among customers who actually churn

**Policy used by the ML model:**
`policy_ml = P(churn) * value_pos`

Where `value_pos` is a pre-T value proxy available at decision time (I use trailing revenue, `rev_sum_90d`, clipped at 0).

**Why this matters:**
A model can have a decent AUC and still be useless if it can’t prioritize the right customers under budget constraints. VaR@K is directly tied to the action you’d take in a real workflow (limited outreach).

## Baselines (so ML has to earn its keep)
Before training XGBoost, I include baselines that represent what teams often do in practice:

* **Random targeting** (control)
* **Recency-based risk** (large time gap since previous purchase → higher risk)
* **RFM heuristic risk** (rank-based combination of Recency, Frequency, Monetary)

If the ML policy doesn’t beat these, it’s not useful.

## Feature engineering (point-in-time)
All features are constructed from customer behavior strictly prior to `T`:

* trailing revenue sums: 30d / 90d / 180d
* purchase frequency: 30d / 90d
* revenue volatility (std): 90d
* return behavior: count of negative-revenue events (90d)
* average order value (AOV proxy): 90d
* recency gap: days since previous event (`gap_days_prev`)

I explicitly avoid degenerate features that break under this framing.

## Results (held-out test set)
Using the repo’s current evaluation output (`reports/evaluation/test_results.csv`), the ML policy improves captured value-at-risk compared with the best heuristic baseline:

* **5% budget:** 101,521 vs 75,090 (+35.2%)
* **10% budget:** 164,788 vs 107,171 (+53.8%)
* **20% budget:** 254,018 vs 197,438 (+28.7%)

These are decision improvements under capacity constraints, not just “better AUC.”

## What makes this “productionized” (not just a notebook)
I built the repo so it has a real lifecycle:

### 1) Deterministic pipelines (repeatable artifacts)
* `build_features` writes canonical datasets to `data/processed/`
* `train` writes reports and model artifacts + MLflow run
* `score` reads the promoted run artifact and produces top-K lists + monitoring outputs

### 2) MLflow tracking with reproducibility
Training logs:
* model params
* VaR@K metrics + uplift metrics
* evaluation artifacts (`val_results.csv`, `test_results.csv`)
* a dataset hash (`data_version`) for reproducibility

### 3) Promotion + scoring from the promoted MLflow run
* Training writes: `models/promoted/production.json` containing the promoted `run_id`
* Scoring loads the model using: `runs:/<run_id>/model`
* *So “promotion” actually changes what’s deployed/scored, instead of being a label.*

### 4) Monitoring output each scoring run
Drift report written to: `reports/monitoring/drift_latest.json`
* **Feature drift:** PSI per feature (warn/alert thresholds)
* **Score drift:** reference vs current score quantiles (mean, p50, p90, p99)

### 5) One-command runbook
There is a Windows PowerShell runbook that executes the entire system end-to-end.

## Project structure
```text
config/
  config.yaml

src/churnxgb/
  data/           # load + clean + invoice aggregation
  labeling/       # churn label definition
  features/       # event table + rolling feature engineering
  baselines/      # recency + RFM heuristics
  modeling/       # XGBoost training + MLflow model loading
  policy/         # policy scoring: p(churn) * value_pos
  evaluation/     # VaR@K metrics + reports
  monitoring/     # drift reference + drift report (PSI + score stats)
  pipeline/       # build_features.py, train.py, score.py
  utils/          # hashing (dataset versioning)

reports/
  evaluation/
  monitoring/

outputs/
  predictions/
  targets/

models/
  registry/
  promoted/
```

## How to run

**Option A: one-command runbook (Windows)**
From repo root:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```
Skip rebuilding features:
```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -SkipBuildFeatures
```

**Option B: run step-by-step**
```powershell
$env:PYTHONPATH = "$PWD\src"

python -m churnxgb.pipeline.build_features
python -m churnxgb.pipeline.train
python -m churnxgb.pipeline.score
```

## Key outputs

### Training
* **Evaluation reports:** `reports/evaluation/val_results.csv`, `reports/evaluation/test_results.csv`
* **Drift reference profile:** `reports/monitoring/reference_profile.json`
* **Promotion pointer:** `models/promoted/production.json`
* **MLflow DB:** `mlflow.db`

### Scoring
* **Predictions table:** `outputs/predictions/predictions_all.parquet`
* **Target lists (top-K by policy_ml):** `outputs/targets/targets_all_k05.parquet`, `outputs/targets/targets_all_k10.parquet`, `outputs/targets/targets_all_k20.parquet`
* **Drift report:** `reports/monitoring/drift_latest.json`

## Small implementation notes (why I did it this way)
* **Customer event deduping:** I create a canonical event table at (CustomerID, InvoiceDate) so multiple invoices at the same timestamp don’t inflate rolling features or labels.
* **Decision-first evaluation:** I kept ROC-AUC as a secondary check, but prioritized VaR@K because that’s what drives an actual retention workflow.
* **Value proxy is pre-T:** it’s tempting to use “future 90d revenue” as value, but churners have no purchases by definition. A production system needs a value proxy known at decision time.
* **Promotion is tied to run_id:** a promotion record that only points to a local file is easy to fake. Using `runs:/<run_id>/model` makes the lifecycle credible.
* **Monitoring is lightweight but real:** PSI on features + score quantiles gives a practical signal without turning this into a full platform.

## Next ideas (if I extend this further)
* Add a scheduler (Prefect/Airflow) for recurring train/score runs
* Add delayed-label performance tracking (because churn labels mature after 90 days)
* Add CI tests (leakage checks, schema checks, metric tests)