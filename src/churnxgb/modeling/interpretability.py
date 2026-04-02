from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def _unwrap_model(model):
    return getattr(model, "base_model", model)


def _booster_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame | None:
    model = _unwrap_model(model)
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
    explain_model = _unwrap_model(model)

    try:
        explainer = shap.Explainer(explain_model, X_small)
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
        fi_df = _booster_feature_importance(explain_model, feature_cols)
        if fi_df is None:
            # Fall back to generic feature_importances_ if exposed.
            if hasattr(explain_model, "feature_importances_"):
                fi_df = (
                    pd.DataFrame(
                        {
                            "feature": feature_cols,
                            "importance": np.asarray(explain_model.feature_importances_, dtype=float),
                            "source": "native_feature_importance",
                        }
                    )
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
            elif hasattr(explain_model, "named_steps") and "model" in explain_model.named_steps:
                inner = explain_model.named_steps["model"]
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _is_logistic_pipeline(model) -> bool:
    model = _unwrap_model(model)
    return (
        isinstance(model, Pipeline)
        and hasattr(model, "named_steps")
        and "model" in model.named_steps
        and hasattr(model.named_steps["model"], "coef_")
        and hasattr(model.named_steps["model"], "intercept_")
    )


def _is_tree_model(model) -> bool:
    model = _unwrap_model(model)
    return hasattr(model, "get_booster") or hasattr(model, "booster_")


def _prepare_logistic_transformed_frame(
    model: Pipeline, X: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = _unwrap_model(model)
    X_input = X[feature_cols].copy()
    transformed = X_input.copy()

    if "imputer" in model.named_steps:
        transformed = pd.DataFrame(
            model.named_steps["imputer"].transform(transformed),
            columns=feature_cols,
            index=X_input.index,
        )

    if "scaler" in model.named_steps:
        transformed = pd.DataFrame(
            model.named_steps["scaler"].transform(transformed),
            columns=feature_cols,
            index=X_input.index,
        )

    return X_input, transformed


def _explain_logistic_pipeline_rows(
    model: Pipeline,
    X: pd.DataFrame,
    feature_cols: list[str],
    top_n: int,
) -> list[dict]:
    model = _unwrap_model(model)
    X_input, transformed = _prepare_logistic_transformed_frame(model, X, feature_cols)
    lr = model.named_steps["model"]
    coef = np.asarray(lr.coef_[0], dtype=float)
    intercept = float(lr.intercept_[0])

    logits = transformed.to_numpy(dtype=float) @ coef + intercept
    probs = _sigmoid(logits)
    base_prob = float(_sigmoid(np.array([intercept]))[0])

    rows: list[dict] = []
    for row_idx, idx in enumerate(X_input.index):
        contrib = transformed.iloc[row_idx].to_numpy(dtype=float) * coef
        contrib_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "feature_value": X_input.iloc[row_idx].to_numpy(dtype=float),
                "transformed_value": transformed.iloc[row_idx].to_numpy(dtype=float),
                "coefficient": coef,
                "contribution_logit": contrib,
            }
        ).sort_values("contribution_logit", ascending=False)

        top_positive = contrib_df[contrib_df["contribution_logit"] > 0].head(top_n)
        top_negative = contrib_df.sort_values("contribution_logit").head(top_n)
        top_negative = top_negative[top_negative["contribution_logit"] < 0]

        rows.append(
            {
                "row_index": int(row_idx),
                "input_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "explanation_method": "logistic_pipeline_logit_contributions",
                "base_value_logit": intercept,
                "base_value_probability": base_prob,
                "prediction_logit": float(logits[row_idx]),
                "prediction_probability": float(probs[row_idx]),
                "feature_contributions": contrib_df.to_dict(orient="records"),
                "top_positive_contributors": top_positive.to_dict(orient="records"),
                "top_negative_contributors": top_negative.to_dict(orient="records"),
            }
        )

    return rows


def _explain_with_shap(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    top_n: int,
) -> list[dict]:
    model = _unwrap_model(model)
    X_input = X[feature_cols].copy()
    background = X_input.head(min(len(X_input), 200)).copy()
    explainer = shap.Explainer(model, background)
    shap_values = explainer(X_input)

    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim > 1:
        base_values = base_values[:, -1]

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, -1]

    rows: list[dict] = []
    for row_idx, idx in enumerate(X_input.index):
        contrib_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "feature_value": X_input.iloc[row_idx].to_numpy(dtype=float),
                "shap_value": values[row_idx].astype(float),
            }
        )
        top_positive = contrib_df.sort_values("shap_value", ascending=False)
        top_positive = top_positive[top_positive["shap_value"] > 0].head(top_n)
        top_negative = contrib_df.sort_values("shap_value").head(top_n)
        top_negative = top_negative[top_negative["shap_value"] < 0]

        row_prob = float(model.predict_proba(X_input.iloc[[row_idx]])[:, 1][0])
        rows.append(
            {
                "row_index": int(row_idx),
                "input_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "explanation_method": "shap_row_explanation",
                "base_value": float(base_values[row_idx]),
                "prediction_probability": row_prob,
                "feature_contributions": contrib_df.to_dict(orient="records"),
                "top_positive_contributors": top_positive.to_dict(orient="records"),
                "top_negative_contributors": top_negative.to_dict(orient="records"),
            }
        )
    return rows


def _explain_tree_model_rows(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    top_n: int,
) -> list[dict]:
    model = _unwrap_model(model)
    X_input = X[feature_cols].copy()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[-1], dtype=float)
    else:
        values = np.asarray(shap_values, dtype=float)

    base_values = explainer.expected_value
    if isinstance(base_values, (list, tuple, np.ndarray)):
        base_array = np.asarray(base_values, dtype=float)
        if base_array.ndim == 0:
            base_array = np.repeat(base_array.item(), len(X_input))
        else:
            base_array = np.repeat(base_array.ravel()[-1], len(X_input))
    else:
        base_array = np.repeat(float(base_values), len(X_input))

    rows: list[dict] = []
    for row_idx, idx in enumerate(X_input.index):
        contrib_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "feature_value": X_input.iloc[row_idx].to_numpy(dtype=float),
                "shap_value": values[row_idx].astype(float),
            }
        )
        top_positive = contrib_df.sort_values("shap_value", ascending=False)
        top_positive = top_positive[top_positive["shap_value"] > 0].head(top_n)
        top_negative = contrib_df.sort_values("shap_value").head(top_n)
        top_negative = top_negative[top_negative["shap_value"] < 0]

        row_prob = float(model.predict_proba(X_input.iloc[[row_idx]])[:, 1][0])
        rows.append(
            {
                "row_index": int(row_idx),
                "input_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "explanation_method": "tree_shap_row_explanation",
                "base_value": float(base_array[row_idx]),
                "prediction_probability": row_prob,
                "feature_contributions": contrib_df.to_dict(orient="records"),
                "top_positive_contributors": top_positive.to_dict(orient="records"),
                "top_negative_contributors": top_negative.to_dict(orient="records"),
            }
        )
    return rows


def explain_prediction_rows(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Return row-level explanations for prediction inputs.

    For the current promoted logistic-regression pipeline, explanations are exact
    logit-space feature contributions from the fitted standardized linear model.
    Because the pipeline uses a StandardScaler, the baseline intercept corresponds
    to the standardized zero point, which maps back to the training-data mean
    after scaling rather than raw feature zeros.
    For other model families, the function falls back to SHAP-based row summaries.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    X_input = X[feature_cols].copy()
    if _is_logistic_pipeline(model):
        return _explain_logistic_pipeline_rows(model, X_input, feature_cols, top_n)
    if _is_tree_model(model):
        return _explain_tree_model_rows(model, X_input, feature_cols, top_n)

    return _explain_with_shap(model, X_input, feature_cols, top_n)
