"""Plotting helpers for project reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot counts of the binary target."""
    counts = df["depression"].value_counts().sort_index()
    labels = [str(label) for label in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts.values)
    ax.set_title("Target distribution")
    ax.set_xlabel("Depression label")
    ax.set_ylabel("Count")

    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, output_path: str | Path) -> None:
    """Plot missing-value counts after cleaning."""
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values after cleaning", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(missing.index.astype(str), missing.values)
        ax.set_title("Missing values after cleaning")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame, output_path: str | Path, max_features: int = 12) -> None:
    """Plot a correlation heatmap for selected numeric variables."""
    numeric = df.select_dtypes(include="number")
    if "depression" in numeric.columns:
        candidate_columns = [col for col in numeric.columns if col != "depression"]
    else:
        candidate_columns = list(numeric.columns)

    if len(candidate_columns) < 2:
        return

    selected = candidate_columns[:max_features]
    corr = numeric[selected].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Correlation among selected numeric features")
    ax.set_xticks(np.arange(len(selected)))
    ax.set_yticks(np.arange(len(selected)))
    ax.set_xticklabels(selected, rotation=45, ha="right")
    ax.set_yticklabels(selected)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_depression_rate_by_feature(df: pd.DataFrame, feature: str, output_path: str | Path, top_n: int = 20) -> None:
    """Plot target rate by a selected feature."""
    if feature not in df.columns:
        return

    table = (
        df.groupby(feature, dropna=False)["depression"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(table.index.astype(str), table.values)
    ax.set_title(f"Depression label rate by {feature}")
    ax.set_xlabel("Depression label rate")
    ax.set_xlim(0, min(1.0, max(table.values) * 1.15 if len(table) else 1.0))

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(model_results: dict, output_path: str | Path) -> None:
    """Plot ROC curves for all fitted models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in model_results.items():
        RocCurveDisplay.from_predictions(result["y_test"], result["y_score"], name=name, ax=ax)
    ax.set_title("ROC curve comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curves(model_results: dict, output_path: str | Path) -> None:
    """Plot precision-recall curves for all fitted models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in model_results.items():
        PrecisionRecallDisplay.from_predictions(result["y_test"], result["y_score"], name=name, ax=ax)
    ax.set_title("Precision-Recall curve comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str | Path) -> None:
    """Plot a confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_feature_bar(
    table: pd.DataFrame,
    feature_col: str,
    value_col: str,
    title: str,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Plot top feature importance or coefficient values."""
    if table.empty or value_col not in table.columns:
        return

    data = table.copy()
    data["_abs"] = data[value_col].abs()
    data = data.sort_values("_abs", ascending=False).head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data[feature_col].astype(str), data[value_col])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
