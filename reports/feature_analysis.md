# Feature Analysis

Primary model: `logistic_regression`

Importance method: Native feature importance fallback

Top features reflect the behavioral signals driving churn prioritization.

| feature          |   importance | source            |
|:-----------------|-------------:|:------------------|
| freq_90d         |     1.88788  | abs_logistic_coef |
| rev_sum_180d     |     1.29013  | abs_logistic_coef |
| rev_sum_90d      |     0.941079 | abs_logistic_coef |
| rev_sum_30d      |     0.931019 | abs_logistic_coef |
| return_count_90d |     0.26653  | abs_logistic_coef |
| gap_days_prev    |     0.258065 | abs_logistic_coef |
| freq_30d         |     0.210897 | abs_logistic_coef |
| rev_std_90d      |     0.143123 | abs_logistic_coef |
| aov_90d          |     0.07502  | abs_logistic_coef |
