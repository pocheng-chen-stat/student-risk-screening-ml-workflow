"""Plotting helpers for project reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_target_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="depression", ax=ax)
    ax.set_title("Target distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, output_path: str | Path) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values after cleaning", ha="center", va="center")
        ax.axis("off")
    else:
        missing.plot(kind="bar", ax=ax)
        ax.set_title("Missing values after cleaning")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame, output_path: str | Path) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric correlation heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_depression_rate_by_feature(df: pd.DataFrame, feature: str, output_path: str | Path, top_n: int = 20) -> None:
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
    table.plot(kind="barh", ax=ax)
    ax.set_title(f"Depression rate by {feature}")
    ax.set_xlabel("Depression rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(model_results: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in model_results.items():
        RocCurveDisplay.from_predictions(result["y_test"], result["y_score"], name=name, ax=ax)
    ax.set_title("ROC curve comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curves(model_results: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in model_results.items():
        PrecisionRecallDisplay.from_predictions(result["y_test"], result["y_score"], name=name, ax=ax)
    ax.set_title("Precision-Recall curve comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title: str, output_path: str | Path) -> None:
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
