from __future__ import annotations

import pandas as pd

from churnxgb.evaluation.experiment_simulation import (
    get_experiment_config,
    simulate_experiment_by_budget,
)


def test_simulate_experiment_by_budget_returns_assumption_driven_outputs() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4],
            "policy_net_benefit": [12.0, 10.0, 8.0, 1.0],
            "churn_prob": [0.8, 0.6, 0.4, 0.2],
            "value_pos": [100.0, 80.0, 60.0, 20.0],
            "assumed_success_rate_customer": [0.2, 0.15, 0.1, 0.05],
            "expected_retained_value": [16.0, 7.2, 2.4, 0.2],
        }
    )
    cfg = get_experiment_config(
        {
            "experiment": {
                "treatment_allocation_rate": 0.5,
                "treatment_effect_multiplier": 1.0,
                "targeting_policy": "policy_net_benefit",
            }
        }
    )

    result = simulate_experiment_by_budget(df, [0.5], cfg)

    assert len(result) == 1
    assert result[0]["assumption_driven"] is True
    assert result[0]["targeted_customers"] == 2
    assert result[0]["treatment_customers"] == 1
    assert result[0]["control_customers"] == 1
    assert "incremental_retained_value" in result[0]
    assert "incremental_retained_value_ci" not in result[0]
    assert "uplift_probability_ci" not in result[0]
