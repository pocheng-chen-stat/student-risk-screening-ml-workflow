"""Markdown report helpers for README-ready results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "model",
    "accuracy",
    "roc_auc",
    "pr_auc",
    "sensitivity_recall",
    "specificity",
    "precision",
    "f1",
]


def _format_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in METRIC_COLUMNS if column in table.columns]
    formatted = table.loc[:, columns].copy()
    for column in formatted.columns:
        if column != "model":
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted


def _top_features(table: pd.DataFrame, value_col: str, top_n: int = 10) -> pd.DataFrame:
    if table.empty or value_col not in table.columns:
        return pd.DataFrame()
    data = table.copy()
    data["_abs"] = data[value_col].abs()
    data = data.sort_values("_abs", ascending=False).head(top_n)
    output = data[["feature", value_col]].copy()
    output[value_col] = output[value_col].map(lambda value: f"{value:.4f}")
    return output


def save_readme_results(
    *,
    test_comparison: pd.DataFrame,
    cv_comparison: pd.DataFrame,
    interpretation_outputs: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("## Model Performance")
    lines.append("")
    lines.append("Held-out test performance after training-set cross-validation tuning:")
    lines.append("")
    lines.append(_format_metric_table(test_comparison).to_markdown(index=False))
    lines.append("")
    lines.append("Training-set cross-validation stability:")
    lines.append("")
    cv_columns = ["model", "best_cv_roc_auc", "cv_std_roc_auc", "best_params"]
    available_cv_columns = [column for column in cv_columns if column in cv_comparison.columns]
    formatted_cv = cv_comparison.loc[:, available_cv_columns].copy()
    for column in ["best_cv_roc_auc", "cv_std_roc_auc"]:
        if column in formatted_cv.columns:
            formatted_cv[column] = formatted_cv[column].map(lambda value: f"{value:.3f}")
    lines.append(formatted_cv.to_markdown(index=False))
    lines.append("")
    lines.append("![ROC curve comparison](reports/figures/roc_curve_comparison.png)")
    lines.append("")
    lines.append("![Precision-Recall curve comparison](reports/figures/precision_recall_curve_comparison.png)")
    lines.append("")
    lines.append("## Important Predictive Features")
    lines.append("")
    lines.append("Feature importance should be interpreted as predictive signal, not causal effect.")
    lines.append("")

    feature_sections = [
        ("Lasso logistic coefficients", "lasso_coefficients", "coefficient"),
        ("Random Forest feature importance", "random_forest_importance", "importance"),
        ("XGBoost feature importance", "xgboost_importance", "importance"),
        ("Permutation importance", "permutation_importance", "importance_mean"),
    ]
    for title, key, value_col in feature_sections:
        table = interpretation_outputs.get(key)
        if table is None:
            continue
        top_table = _top_features(table, value_col)
        if top_table.empty:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append(top_table.to_markdown(index=False))
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
