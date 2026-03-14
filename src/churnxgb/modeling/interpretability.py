from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _booster_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        rows = []
        for idx, col in enumerate(feature_cols):
            rows.append(
                {
                    "feature": col,
                    "importance": float(score.get(f"f{idx}", 0.0)),
                    "source": "xgboost_gain",
                }
            )
        return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(
            drop=True
        )
    return None


def save_feature_importance_artifacts(
    model,
    X_sample: pd.DataFrame,
    feature_cols: list[str],
    figures_dir: Path,
    out_csv: Path,
) -> tuple[pd.DataFrame, str, Path]:
    """
    Save SHAP summary artifacts when supported, otherwise fall back to model-native importance.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Keep SHAP runtime reasonable for portfolio use.
    X_small = X_sample[feature_cols].copy().head(1000)

    try:
        explainer = shap.Explainer(model, X_small)
        shap_values = explainer(X_small)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        fi_df = (
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "importance": mean_abs.astype(float),
                    "source": "shap_mean_abs",
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.to_csv(out_csv, index=False)

        shap.summary_plot(shap_values, X_small, show=False, plot_type="bar")
        plt.tight_layout()
        shap_path = figures_dir / "shap_summary_bar.png"
        plt.savefig(shap_path, dpi=140, bbox_inches="tight")
        plt.close()
        return fi_df, "SHAP mean absolute importance", shap_path
    except Exception:
        fi_df = _booster_feature_importance(model, feature_cols)
        if fi_df is None:
            # Fall back to generic feature_importances_ if exposed.
            if hasattr(model, "feature_importances_"):
                fi_df = (
                    pd.DataFrame(
                        {
                            "feature": feature_cols,
                            "importance": np.asarray(model.feature_importances_, dtype=float),
                            "source": "native_feature_importance",
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
            elif hasattr(model, "named_steps") and "model" in model.named_steps:
                inner = model.named_steps["model"]
                coefs = getattr(inner, "coef_", None)
                if coefs is not None:
                    fi_df = (
                        pd.DataFrame(
                            {
                                "feature": feature_cols,
                                "importance": np.abs(np.ravel(coefs)).astype(float),
                                "source": "abs_logistic_coef",
                            }
                        )
                        .sort_values("importance", ascending=False)
                        .reset_index(drop=True)
                    )
                else:
                    raise
            else:
                raise

        fi_df.to_csv(out_csv, index=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        top = fi_df.head(15).iloc[::-1]
        ax.barh(top["feature"], top["importance"])
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        fallback_path = figures_dir / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(fallback_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return fi_df, "Native feature importance fallback", fallback_path
