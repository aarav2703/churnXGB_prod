from __future__ import annotations

import pandas as pd


DEFAULT_DECISION_POLICY = {
    "intervention_cost": 0.0,
    "assumed_success_rate": 1.0,
    "retention_value_multiplier": 1.0,
    "targeting_policy": "policy_net_benefit",
    "heterogeneity": {
        "enabled": False,
        "success_rate_recency_weight": 0.0,
        "success_rate_frequency_weight": 0.0,
        "cost_value_weight": 0.0,
        "cost_frequency_weight": 0.0,
        "min_success_rate_multiplier": 1.0,
        "max_success_rate_multiplier": 1.0,
        "min_cost_multiplier": 1.0,
        "max_cost_multiplier": 1.0,
    },
}


def get_decision_policy_config(cfg: dict | None = None) -> dict:
    decision_cfg = (cfg or {}).get("decision", {})
    heterogeneity_cfg = decision_cfg.get("heterogeneity", {})
    out = {
        "intervention_cost": float(
            decision_cfg.get(
                "intervention_cost", DEFAULT_DECISION_POLICY["intervention_cost"]
            )
        ),
        "assumed_success_rate": float(
            decision_cfg.get(
                "assumed_success_rate",
                DEFAULT_DECISION_POLICY["assumed_success_rate"],
            )
        ),
        "retention_value_multiplier": float(
            decision_cfg.get(
                "retention_value_multiplier",
                DEFAULT_DECISION_POLICY["retention_value_multiplier"],
            )
        ),
        "targeting_policy": str(
            decision_cfg.get(
                "targeting_policy",
                DEFAULT_DECISION_POLICY["targeting_policy"],
            )
        ),
        "heterogeneity": {
            "enabled": bool(
                heterogeneity_cfg.get(
                    "enabled",
                    DEFAULT_DECISION_POLICY["heterogeneity"]["enabled"],
                )
            ),
            "success_rate_recency_weight": float(
                heterogeneity_cfg.get(
                    "success_rate_recency_weight",
                    DEFAULT_DECISION_POLICY["heterogeneity"][
                        "success_rate_recency_weight"
                    ],
                )
            ),
            "success_rate_frequency_weight": float(
                heterogeneity_cfg.get(
                    "success_rate_frequency_weight",
                    DEFAULT_DECISION_POLICY["heterogeneity"][
                        "success_rate_frequency_weight"
                    ],
                )
            ),
            "cost_value_weight": float(
                heterogeneity_cfg.get(
                    "cost_value_weight",
                    DEFAULT_DECISION_POLICY["heterogeneity"]["cost_value_weight"],
                )
            ),
            "cost_frequency_weight": float(
                heterogeneity_cfg.get(
                    "cost_frequency_weight",
                    DEFAULT_DECISION_POLICY["heterogeneity"]["cost_frequency_weight"],
                )
            ),
            "min_success_rate_multiplier": float(
                heterogeneity_cfg.get(
                    "min_success_rate_multiplier",
                    DEFAULT_DECISION_POLICY["heterogeneity"][
                        "min_success_rate_multiplier"
                    ],
                )
            ),
            "max_success_rate_multiplier": float(
                heterogeneity_cfg.get(
                    "max_success_rate_multiplier",
                    DEFAULT_DECISION_POLICY["heterogeneity"][
                        "max_success_rate_multiplier"
                    ],
                )
            ),
            "min_cost_multiplier": float(
                heterogeneity_cfg.get(
                    "min_cost_multiplier",
                    DEFAULT_DECISION_POLICY["heterogeneity"]["min_cost_multiplier"],
                )
            ),
            "max_cost_multiplier": float(
                heterogeneity_cfg.get(
                    "max_cost_multiplier",
                    DEFAULT_DECISION_POLICY["heterogeneity"]["max_cost_multiplier"],
                )
            ),
        },
    }
    if out["assumed_success_rate"] < 0:
        raise ValueError("assumed_success_rate must be >= 0.")
    if out["retention_value_multiplier"] < 0:
        raise ValueError("retention_value_multiplier must be >= 0.")
    return out


def get_targeting_policy_name(decision_cfg: dict | None = None) -> str:
    cfg = decision_cfg or DEFAULT_DECISION_POLICY
    return str(cfg.get("targeting_policy", DEFAULT_DECISION_POLICY["targeting_policy"]))


def add_value_pos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "rev_sum_90d" not in out.columns:
        raise ValueError("Expected rev_sum_90d to compute value_pos (value proxy).")
    out["value_proxy"] = out["rev_sum_90d"]
    out["value_pos"] = out["value_proxy"].clip(lower=0.0)
    return out


def _minmax_series(s: pd.Series) -> pd.Series:
    s_num = s.astype(float)
    s_min = float(s_num.min())
    s_max = float(s_num.max())
    if s_max <= s_min:
        return pd.Series(0.0, index=s.index, dtype=float)
    return (s_num - s_min) / (s_max - s_min)


def _apply_customer_level_assumptions(out: pd.DataFrame, sim_cfg: dict) -> pd.DataFrame:
    heterogeneity_cfg = sim_cfg.get("heterogeneity", {})
    enabled = bool(heterogeneity_cfg.get("enabled", False))

    base_success_rate = float(sim_cfg["assumed_success_rate"])
    base_cost = float(sim_cfg["intervention_cost"])

    if not enabled:
        out["assumed_success_rate_customer"] = base_success_rate
        out["intervention_cost_customer"] = base_cost
        return out

    recency_signal = (
        out["recency_risk"].astype(float)
        if "recency_risk" in out.columns
        else pd.Series(0.0, index=out.index, dtype=float)
    )
    frequency_signal = (
        _minmax_series(out["freq_90d"])
        if "freq_90d" in out.columns
        else pd.Series(0.0, index=out.index, dtype=float)
    )
    value_signal = _minmax_series(out["value_pos"])

    success_multiplier = (
        1.0
        + float(heterogeneity_cfg["success_rate_recency_weight"]) * recency_signal
        - float(heterogeneity_cfg["success_rate_frequency_weight"]) * frequency_signal
    ).clip(
        lower=float(heterogeneity_cfg["min_success_rate_multiplier"]),
        upper=float(heterogeneity_cfg["max_success_rate_multiplier"]),
    )
    cost_multiplier = (
        1.0
        + float(heterogeneity_cfg["cost_value_weight"]) * value_signal
        + float(heterogeneity_cfg["cost_frequency_weight"]) * frequency_signal
    ).clip(
        lower=float(heterogeneity_cfg["min_cost_multiplier"]),
        upper=float(heterogeneity_cfg["max_cost_multiplier"]),
    )

    out["assumed_success_rate_customer"] = base_success_rate * success_multiplier
    out["intervention_cost_customer"] = base_cost * cost_multiplier
    return out


def add_policy_scores(df: pd.DataFrame, decision_cfg: dict | None = None) -> pd.DataFrame:
    out = add_value_pos(df)
    sim_cfg = get_decision_policy_config({"decision": (decision_cfg or {})})
    out = _apply_customer_level_assumptions(out, sim_cfg)

    if "churn_prob" in out.columns:
        out["policy_ml"] = out["churn_prob"] * out["value_pos"]
        out["expected_retained_value"] = (
            out["churn_prob"]
            * out["value_pos"]
            * out["assumed_success_rate_customer"]
            * float(sim_cfg["retention_value_multiplier"])
        )
        out["expected_cost"] = out["intervention_cost_customer"]
        out["policy_net_benefit"] = out["expected_retained_value"] - out["expected_cost"]

    if "recency_risk" in out.columns:
        out["policy_recency"] = out["recency_risk"] * out["value_pos"]

    if "rfm_risk" in out.columns:
        out["policy_rfm"] = out["rfm_risk"] * out["value_pos"]

    out["decision_simulation_assumption_driven"] = True
    out["decision_targeting_policy"] = get_targeting_policy_name(sim_cfg)
    return out
