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


def plot_budget_frontier(frontier_df: pd.DataFrame, out_path: Path, title: str) -> Path:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(frontier_df["budget_k"] * 100, frontier_df["value_at_risk"], marker="o", label="VaR@K")
    ax1.set_xlabel("Budget (% targeted)")
    ax1.set_ylabel("Value at Risk")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    if "net_benefit_at_k" in frontier_df.columns:
        ax2.plot(
            frontier_df["budget_k"] * 100,
            frontier_df["net_benefit_at_k"],
            marker="s",
            color="#b86825",
            label="Net Benefit@K",
        )
        ax2.set_ylabel("Net Benefit")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    return _savefig(fig, out_path)


def plot_backtest_trend(
    detail_df: pd.DataFrame,
    out_path: Path,
    title: str,
    value_col: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for model_name, model_df in detail_df.groupby("model"):
        ordered = model_df.sort_values("fold")
        ax.plot(ordered["fold"], ordered[value_col], marker="o", label=model_name)
    ax.set_xlabel("Backtest Fold")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return _savefig(fig, out_path)
