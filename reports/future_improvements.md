# Future Improvements

## Highest-Value Next Steps

### 1. Add targeted hyperparameter search

Current comparison uses one configuration per model family. A lightweight temporal tuning loop would strengthen the experimentation story without overengineering the repo.

Practical next step:
- search a small grid for XGBoost and LightGBM over depth, learning rate, regularization, and number of estimators
- score candidates on validation VaR@10 and PR-AUC

Why it helps:
- stronger signal for experimentation rigor
- better evidence that the chosen model was selected deliberately

### 2. Add explicit probability calibration

The project now tracks Brier score and calibration plots, but it does not yet fit calibrated probability models.

Practical next step:
- compare raw probabilities against Platt scaling and isotonic calibration on the validation split
- report whether calibration improves Brier score and policy ranking stability

Why it helps:
- stronger applied ML signal around reliable probabilities
- makes score thresholds and business interpretation more defensible

### 3. Add cohort error analysis

The repo now compares models overall, but not by segment.

Practical next step:
- slice performance by month, revenue band, purchase frequency band, and new vs returning customer cohorts
- write a compact error analysis report and dashboard view

Why it helps:
- shows product and analytics maturity
- helps explain when the model works well or poorly

### 4. Add delayed-label performance tracking

The current score pipeline writes predictions and drift outputs, but it does not yet reconcile predictions with matured future labels later.

Practical next step:
- create a delayed evaluation script that joins past scored cohorts with matured churn outcomes after 90 days
- log realized Precision@K, Recall@K, and VaR capture over time

Why it helps:
- strengthens the “real ML lifecycle” story
- shows awareness of deployment-time label latency

### 5. Add alert-oriented monitoring summaries

Monitoring currently computes PSI and score distribution stats, which is useful, but it does not surface actionable alert summaries beyond raw counts.

Practical next step:
- add simple severity flags and markdown summaries for drift
- highlight top shifted features and score shifts directly in the dashboard

Why it helps:
- improves recruiter-facing MLOps polish
- makes the monitoring outputs easier to interpret quickly

### 6. Improve dashboard UX for storytelling

The dashboard works and covers the key panels, but it can become more persuasive for reviewers.

Practical next step:
- add short explanatory captions around each plot
- add model selector and fold selector
- surface the promoted run id, chosen budget, and uplift headline at the top

Why it helps:
- improves demo quality for interviews and portfolio reviews
- makes the project feel more intentional and product-aware

### 7. Tighten environment reproducibility

`requirements.txt` is present, but the local environment still showed package compatibility warnings during validation.

Practical next step:
- pin `protobuf` explicitly to a Streamlit-compatible version
- add a short environment setup note using a clean virtual environment
- optionally add a lockfile or exported conda environment

Why it helps:
- reduces setup friction for reviewers
- strengthens the reliability/reproducibility signal

## Lower-Priority Extensions

- add treatment-response simulation assumptions to translate VaR capture into hypothetical retained revenue
- add feature ablation analysis to show which feature groups matter most
- expand the dataset or add a second churn dataset to demonstrate portability

## What Not To Prioritize Next

The project does **not** currently need heavyweight infrastructure such as Kubernetes, distributed serving, or a complex feature store. The highest return now is stronger evaluation depth, calibration, segmentation analysis, and cleaner reproducibility.
