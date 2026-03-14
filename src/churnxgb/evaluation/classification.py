from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve


def classification_summary(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    """
    Core probabilistic classification metrics for churn prediction.
    """
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(y_prob).astype(float)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(y_true.mean()),
    }


def curve_data(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> dict[str, pd.DataFrame]:
    """
    Curve data for plotting and dashboard consumption.
    """
    y_true = pd.Series(y_true).astype(int)
    y_prob = pd.Series(y_prob).astype(float)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    roc_df = pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "threshold": np.append(roc_thresholds, np.nan)[: len(fpr)]}
    )
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan)[: len(precision)],
        }
    )
    calib_df = pd.DataFrame(
        {"mean_predicted_probability": mean_pred, "observed_positive_rate": frac_pos}
    )

    return {"roc": roc_df, "pr": pr_df, "calibration": calib_df}


def save_curve_data(
    curve_frames: dict[str, pd.DataFrame],
    out_dir: Path,
    split_name: str,
    model_name: str,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for curve_name, frame in curve_frames.items():
        out_path = out_dir / f"{model_name}_{split_name}_{curve_name}.csv"
        frame.to_csv(out_path, index=False)
        saved[curve_name] = out_path
    return saved
