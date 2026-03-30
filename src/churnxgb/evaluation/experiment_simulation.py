from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_EXPERIMENT_CONFIG = {
    "treatment_allocation_rate": 0.5,
    "treatment_effect_multiplier": 1.0,
    "targeting_policy": "policy_net_benefit",
}


def get_experiment_config(cfg: dict | None = None) -> dict[str, float | str]:
    experiment_cfg = (cfg or {}).get("experiment", {})
    out = {
        "treatment_allocation_rate": float(
            experiment_cfg.get(
                "treatment_allocation_rate",
                DEFAULT_EXPERIMENT_CONFIG["treatment_allocation_rate"],
            )
        ),
        "treatment_effect_multiplier": float(
            experiment_cfg.get(
                "treatment_effect_multiplier",
                DEFAULT_EXPERIMENT_CONFIG["treatment_effect_multiplier"],
            )
        ),
        "targeting_policy": str(
            experiment_cfg.get(
                "targeting_policy", DEFAULT_EXPERIMENT_CONFIG["targeting_policy"]
            )
        ),
    }
    if not (0.0 < out["treatment_allocation_rate"] < 1.0):
        raise ValueError("treatment_allocation_rate must be in (0, 1).")
    if out["treatment_effect_multiplier"] < 0.0:
        raise ValueError("treatment_effect_multiplier must be >= 0.")
    return out


@dataclass(frozen=True)
class ExperimentInputs:
    treatment_allocation_rate: float
    treatment_effect_multiplier: float
    targeting_policy: str


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required experiment simulation columns: {missing}")


def _prepare_simulation_frame(
    scored_df: pd.DataFrame, targeting_policy: str, treatment_effect_multiplier: float
) -> pd.DataFrame:
    required = [
        "CustomerID",
        "churn_prob",
        "value_pos",
        "assumed_success_rate_customer",
        "expected_retained_value",
        targeting_policy,
    ]
    _require_columns(scored_df, required)

    df = scored_df.copy()
    df["simulated_uplift_probability"] = (
        df["churn_prob"].astype(float)
        * df["assumed_success_rate_customer"].astype(float)
        * float(treatment_effect_multiplier)
    ).clip(lower=0.0, upper=1.0)
    df["simulated_incremental_retained_value"] = (
        df["simulated_uplift_probability"].astype(float)
        * df["value_pos"].astype(float)
    )
    return df


def simulate_experiment_by_budget(
    scored_df: pd.DataFrame,
    budgets: list[float],
    experiment_cfg: dict[str, float | str],
) -> list[dict]:
    cfg = ExperimentInputs(
        treatment_allocation_rate=float(experiment_cfg["treatment_allocation_rate"]),
        treatment_effect_multiplier=float(experiment_cfg["treatment_effect_multiplier"]),
        targeting_policy=str(experiment_cfg["targeting_policy"]),
    )
    df = _prepare_simulation_frame(
        scored_df,
        targeting_policy=cfg.targeting_policy,
        treatment_effect_multiplier=cfg.treatment_effect_multiplier,
    )

    results: list[dict] = []
    for k in budgets:
        if not (0 < k <= 1):
            raise ValueError("budget k must be in (0, 1].")

        top_n = max(1, int(round(len(df) * float(k))))
        targeted = df.sort_values(cfg.targeting_policy, ascending=False).head(top_n).copy()
        targeted = targeted.reset_index(drop=True)

        treat_n = max(1, int(round(len(targeted) * cfg.treatment_allocation_rate)))
        treatment = targeted.iloc[:treat_n].copy()
        control = targeted.iloc[treat_n:].copy()

        if len(control) == 0:
            control = targeted.iloc[0:0].copy()

        treatment["experiment_arm"] = "treatment"
        control["experiment_arm"] = "control"

        delta = treatment["simulated_incremental_retained_value"].astype(float)
        uplift = treatment["simulated_uplift_probability"].astype(float)
        avg_delta = float(delta.mean()) if len(delta) else 0.0
        avg_uplift = float(uplift.mean()) if len(uplift) else 0.0
        total_delta = float(delta.sum()) if len(delta) else 0.0
        total_control = 0.0

        results.append(
            {
                "budget_k": float(k),
                "assumption_driven": True,
                "targeting_policy": cfg.targeting_policy,
                "treatment_allocation_rate": cfg.treatment_allocation_rate,
                "treatment_effect_multiplier": cfg.treatment_effect_multiplier,
                "targeted_customers": int(len(targeted)),
                "treatment_customers": int(len(treatment)),
                "control_customers": int(len(control)),
                "treatment_expected_incremental_retained_value": total_delta,
                "control_expected_incremental_retained_value": total_control,
                "incremental_retained_value": total_delta - total_control,
                "average_incremental_retained_value_per_treated_customer": avg_delta,
                "average_uplift_probability_treatment": avg_uplift,
                "uplift_style_difference_vs_control": avg_uplift,
                "top_treated_customer_ids": treatment["CustomerID"].astype(str).head(20).tolist(),
                "top_control_customer_ids": control["CustomerID"].astype(str).head(20).tolist(),
                "limitation_note": "This is a deterministic business-case simulation using fixed treatment assumptions on model outputs. It does not estimate causal uplift or statistical uncertainty from observed experimental outcomes.",
            }
        )
    return results
