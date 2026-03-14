# Model Evaluation Summary

Best promoted model: `logistic_regression` selected by validation `value_at_risk` at 10% budget.

## Test Classification Metrics

- ROC-AUC: 0.7335
- PR-AUC: 0.5819
- Brier score: 0.2016
- Positive rate: 0.3793

## Test Targeting Metrics (policy_ml)

|   budget_k |   value_at_risk |   var_covered_frac |   precision_at_k |   recall_at_k |   lift_at_k |   captured_churners |
|-----------:|----------------:|-------------------:|-----------------:|--------------:|------------:|--------------------:|
|       0.05 |          108496 |           0.173828 |         0.460432 |     0.0607211 |     1.21398 |                  64 |
|       0.1  |          165919 |           0.265829 |         0.406475 |     0.107211  |     1.07172 |                 113 |
|       0.2  |          260600 |           0.417522 |         0.393885 |     0.20778   |     1.03853 |                 219 |

## Figures

- `reports/figures/test_roc_curve.png`
- `reports/figures/test_pr_curve.png`
- `reports/figures/test_lift_curve.png`
- `reports/figures/test_calibration_curve.png`
