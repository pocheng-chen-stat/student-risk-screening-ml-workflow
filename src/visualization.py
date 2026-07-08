"""Plotting helpers for project reports.

The project uses readable labels in figures because the goal is to communicate
an analysis workflow, not merely expose raw Python column names.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.feature_display import display_feature_label


IMPORTANT_FEATURE_COLORS = {
    "Suicidal Thoughts": "#4C78A8",
    "Academic Pressure": "#F58518",
    "Financial Stress": "#54A24B",
    "Dietary Habits": "#E45756",
    "Age": "#72B7B2",
    "Work/Study Hours": "#B279A2",
    "CGPA": "#FF9DA6",
    "Study Satisfaction": "#9D755D",
    "Sleep Duration": "#BAB0AC",
    "Family History Of Mental Illness": "#59A14F",
}

FALLBACK_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#8CD17D",
    "#B6992D",
    "#499894",
]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _color_for_label(label: str, color_map: dict[str, str]) -> str:
    if label not in color_map:
        color_map[label] = FALLBACK_COLORS[len(color_map) % len(FALLBACK_COLORS)]
    return color_map[label]


def _format_axis_label(label: str) -> str:
    return label.replace("_", " ").title()


def plot_target_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    counts = df["depression"].value_counts().sort_index()
    labels = ["No label (0)", "Risk label (1)"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values)
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height()):,}",
            ha="center",
            va="bottom",
        )
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
        labels = [display_feature_label(feature) for feature in missing.index]
        ax.bar(labels, missing.values)
        ax.set_title("Missing Values After Cleaning")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_numeric_correlation(df: pd.DataFrame, output_path: str | Path) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr()
    labels = [display_feature_label(feature) for feature in corr.columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Numeric Correlation Heatmap")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    labels = [str(label).strip("'").strip('"') for label in table.index]
    ax.barh(labels, table.values)
    ax.set_title(f"Risk Label Rate by {display_feature_label(feature)}")
    ax.set_xlabel("Risk Label Rate")
    ax.set_xlim(0, min(1.0, max(table.values) * 1.15 if len(table) else 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(model_results: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in model_results.items():
        RocCurveDisplay.from_predictions(result["y_test"], result["y_score"], name=_format_axis_label(name), ax=ax)
    ax.set_title("ROC Curve Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curves(model_results: dict, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, result in model_results.items():
        PrecisionRecallDisplay.from_predictions(
            result["y_test"],
            result["y_score"],
            name=_format_axis_label(name),
            ax=ax,
        )
    ax.set_title("Precision-Recall Curve Comparison")
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


def plot_model_metric_comparison(test_comparison: pd.DataFrame, output_path: str | Path) -> None:
    metrics = ["roc_auc", "f1", "sensitivity_recall"]
    metric_labels = ["ROC-AUC", "F1", "Sensitivity"]
    plot_df = test_comparison.set_index("model")[metrics].copy()
    plot_df.index = [_format_axis_label(model) for model in plot_df.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(plot_df.index))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, metric, label in zip(offsets, metrics, metric_labels):
        values = plot_df[metric].values
        positions = [i + offset for i in x]
        ax.bar(positions, values, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Held-Out Test Performance by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_model_comparison_overview(
    *,
    model_results: dict,
    test_comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Create a README-ready three-panel overview of model performance.

    The figure combines:
    1. ROC curves for ranking performance.
    2. Precision-recall curves for positive-class screening performance.
    3. A compact bar chart of ROC-AUC, F1, and sensitivity.

    Individual detailed figures are still generated separately, but this
    overview figure is the one intended for the README so the model comparison
    section reads as one coherent result instead of three disconnected plots.
    """
    metrics = ["roc_auc", "f1", "sensitivity_recall"]
    metric_labels = ["ROC-AUC", "F1", "Sensitivity"]
    plot_df = test_comparison.set_index("model")[metrics].copy()
    plot_df.index = [_format_axis_label(model) for model in plot_df.index]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # Panel A: ROC curve comparison.
    ax = axes[0]
    for name, result in model_results.items():
        RocCurveDisplay.from_predictions(
            result["y_test"],
            result["y_score"],
            name=_format_axis_label(name),
            ax=ax,
        )
    ax.set_title("A. ROC Curve")
    ax.legend(fontsize=8, loc="lower right")

    # Panel B: precision-recall curve comparison.
    ax = axes[1]
    for name, result in model_results.items():
        PrecisionRecallDisplay.from_predictions(
            result["y_test"],
            result["y_score"],
            name=_format_axis_label(name),
            ax=ax,
        )
    ax.set_title("B. Precision-Recall Curve")
    ax.legend(fontsize=8, loc="lower left")

    # Panel C: compact metric summary.
    ax = axes[2]
    x = range(len(plot_df.index))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, metric, label in zip(offsets, metrics, metric_labels):
        values = plot_df[metric].values
        positions = [i + offset for i in x]
        ax.bar(positions, values, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df.index, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("C. Held-Out Metric Summary")
    ax.legend(fontsize=8)

    fig.suptitle("Model Comparison Overview", fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
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
    label_col = "feature_label" if "feature_label" in data.columns else feature_col
    data["_abs"] = data[value_col].abs()
    data = data.sort_values("_abs", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(data[label_col].astype(str), data[value_col])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cross_model_feature_importance(
    *,
    lasso_table: pd.DataFrame,
    random_forest_table: pd.DataFrame,
    xgboost_table: pd.DataFrame | None,
    output_path: str | Path,
    top_n: int = 10,
) -> None:
    """Create the main three-panel feature-importance comparison figure."""
    panels: list[tuple[str, pd.DataFrame, str]] = []
    if lasso_table is not None and not lasso_table.empty:
        panels.append(("Lasso\nabsolute coefficient", lasso_table, "abs_coefficient"))
    if random_forest_table is not None and not random_forest_table.empty:
        panels.append(("Random Forest\nfeature importance", random_forest_table, "importance"))
    if xgboost_table is not None and not xgboost_table.empty:
        panels.append(("XGBoost\nfeature importance", xgboost_table, "importance"))

    if not panels:
        return

    color_map = dict(IMPORTANT_FEATURE_COLORS)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8 * len(panels), 6.2), sharey=False)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, table, value_col) in zip(axes, panels):
        data = table.sort_values(value_col, ascending=False).head(top_n).iloc[::-1].copy()
        labels = data["feature_label"].astype(str).tolist()
        colors = [_color_for_label(label, color_map) for label in labels]
        ax.barh(labels, data[value_col], color=colors)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        ax.tick_params(axis="y", labelsize=9)

        if value_col == "abs_coefficient" and "direction" in data.columns:
            xmax = data[value_col].max() if len(data) else 0
            for idx, (_, row) in enumerate(data.iterrows()):
                direction = "+" if row["direction"] == "Positive" else "−"
                level = row.get("level_label")
                label = f"{direction}"
                if isinstance(level, str) and level:
                    label += f" {level}"
                ax.text(row[value_col] + xmax * 0.02, idx, label, va="center", fontsize=8)
            ax.set_xlim(0, data[value_col].max() * 1.28)

    fig.suptitle("Cross-Model Feature Importance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
