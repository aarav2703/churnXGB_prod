from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _savefig(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curve(roc_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(roc_df["fpr"], roc_df["tpr"], label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    return _savefig(fig, out_path)


def plot_pr_curve(pr_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pr_df["recall"], pr_df["precision"], label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    return _savefig(fig, out_path)


def plot_calibration_curve(calib_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        calib_df["mean_predicted_probability"],
        calib_df["observed_positive_rate"],
        marker="o",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Positive Rate")
    ax.set_title(title)
    ax.legend()
    return _savefig(fig, out_path)


def plot_lift_curve(policy_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(policy_df["budget_k"] * 100, policy_df["lift_at_k"], marker="o")
    ax.axhline(1.0, linestyle="--", color="gray", label="Baseline")
    ax.set_xlabel("Budget (% targeted)")
    ax.set_ylabel("Lift@K")
    ax.set_title(title)
    ax.legend()
    return _savefig(fig, out_path)
