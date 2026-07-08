"""Markdown report helpers for README-ready results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "lasso_logistic": "Lasso Logistic",
    "ridge_logistic": "Ridge Logistic",
    "knn": "KNN",
    "svm_linear": "Linear SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

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

METRIC_RENAMES = {
    "model": "Model",
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "sensitivity_recall": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "f1": "F1",
    "best_cv_roc_auc": "Best CV ROC-AUC",
    "cv_std_roc_auc": "CV Std.",
    "best_params": "Selected Parameters",
}


def model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model).replace("_", " ").title())


def _format_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in METRIC_COLUMNS if column in table.columns]
    formatted = table.loc[:, columns].copy()
    formatted["model"] = formatted["model"].map(model_label)
    for column in formatted.columns:
        if column != "model":
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    return formatted.rename(columns=METRIC_RENAMES)


def _format_cv_table(cv_comparison: pd.DataFrame) -> pd.DataFrame:
    cv_columns = ["model", "best_cv_roc_auc", "cv_std_roc_auc", "best_params"]
    available = [column for column in cv_columns if column in cv_comparison.columns]
    formatted = cv_comparison.loc[:, available].copy()
    if "model" in formatted.columns:
        formatted["model"] = formatted["model"].map(model_label)
    for column in ["best_cv_roc_auc", "cv_std_roc_auc"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
    if "best_params" in formatted.columns:
        formatted["best_params"] = formatted["best_params"].map(lambda value: str(value).replace("model__", ""))
    return formatted.rename(columns=METRIC_RENAMES)


def _top_lasso_features(table: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    data = table.sort_values("abs_coefficient", ascending=False).head(top_n).copy()
    data["Direction"] = data["direction"]
    data["Risk-related Level"] = data["level_label"].fillna("numeric increase")
    data["Abs. Coefficient"] = data["abs_coefficient"].map(lambda value: f"{value:.3f}")
    return data[["feature_label", "Risk-related Level", "Direction", "Abs. Coefficient"]].rename(
        columns={"feature_label": "Feature"}
    )


def _top_importance_features(table: pd.DataFrame, value_col: str, top_n: int = 8) -> pd.DataFrame:
    if table.empty or value_col not in table.columns:
        return pd.DataFrame()
    data = table.sort_values(value_col, ascending=False).head(top_n).copy()
    data["Importance"] = data[value_col].map(lambda value: f"{value:.3f}")
    return data[["feature_label", "Importance"]].rename(columns={"feature_label": "Feature"})


def _feature_consensus(interpretation_outputs: dict[str, pd.DataFrame], top_n: int = 10) -> pd.DataFrame:
    sources = {
        "Lasso": ("lasso_feature_importance", "abs_coefficient"),
        "Random Forest": ("random_forest_feature_importance", "importance"),
        "XGBoost": ("xgboost_feature_importance", "importance"),
    }
    rows = []
    for source, (key, value_col) in sources.items():
        table = interpretation_outputs.get(key)
        if table is None or table.empty:
            continue
        top = table.sort_values(value_col, ascending=False).head(top_n)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            rows.append(
                {
                    "feature": row["feature"],
                    "Feature": row["feature_label"],
                    "Source": source,
                    "Rank": rank,
                }
            )
    if not rows:
        return pd.DataFrame()
    long = pd.DataFrame(rows)
    consensus = (
        long.pivot_table(index=["feature", "Feature"], columns="Source", values="Rank", aggfunc="min")
        .reset_index()
        .fillna("")
    )
    available_sources = [source for source in sources if source in consensus.columns]
    consensus["Models in Top 10"] = consensus[available_sources].apply(lambda row: sum(value != "" for value in row), axis=1)
    for source in available_sources:
        consensus[source] = consensus[source].map(lambda value: "" if value == "" else str(int(value)))
    consensus = consensus.sort_values(["Models in Top 10", "Feature"], ascending=[False, True])
    return consensus.drop(columns=["feature"])


def save_readme_results(
    *,
    test_comparison: pd.DataFrame,
    cv_comparison: pd.DataFrame,
    interpretation_outputs: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = test_comparison.sort_values("roc_auc", ascending=False).iloc[0]
    best_sensitivity = test_comparison.sort_values("sensitivity_recall", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("## Results")
    lines.append("")
    lines.append(
        f"The strongest held-out ROC-AUC is from **{model_label(best_auc['model'])}** "
        f"(ROC-AUC = {best_auc['roc_auc']:.3f}). "
        f"The highest sensitivity is from **{model_label(best_sensitivity['model'])}** "
        f"(sensitivity = {best_sensitivity['sensitivity_recall']:.3f})."
    )
    lines.append("")
    lines.append("### Held-Out Test Performance")
    lines.append("")
    lines.append(_format_metric_table(test_comparison).to_markdown(index=False))
    lines.append("")
    lines.append("![Held-out metric comparison](reports/figures/model_metric_comparison.png)")
    lines.append("")
    lines.append("![ROC curve comparison](reports/figures/roc_curve_comparison.png)")
    lines.append("")
    lines.append("![Precision-Recall curve comparison](reports/figures/precision_recall_curve_comparison.png)")
    lines.append("")
    lines.append("### Training-Set Cross-Validation")
    lines.append("")
    lines.append(_format_cv_table(cv_comparison).to_markdown(index=False))
    lines.append("")
    lines.append("## Cross-Model Feature Interpretation")
    lines.append("")
    lines.append(
        "The goal of this section is not to claim psychological causality. "
        "Instead, the analysis asks whether similar predictive signals appear across "
        "linear and tree-based models."
    )
    lines.append("")
    lines.append("![Cross-model feature importance](reports/figures/cross_model_feature_importance.png)")
    lines.append("")

    lasso = interpretation_outputs.get("lasso_feature_importance", pd.DataFrame())
    if not lasso.empty:
        lines.append("### Lasso Direction Summary")
        lines.append("")
        lines.append(
            "Lasso is useful because it gives both variable magnitude and the direction of the selected signal. "
            "The ranking below uses the absolute coefficient size, while the direction column keeps the sign information."
        )
        lines.append("")
        lines.append(_top_lasso_features(lasso).to_markdown(index=False))
        lines.append("")

    logistic = interpretation_outputs.get("logistic_risk_direction", pd.DataFrame())
    if not logistic.empty:
        pos = logistic[logistic["direction"].eq("Positive")].head(5)
        neg = logistic[logistic["direction"].eq("Negative")].head(5)
        lines.append("### Logistic Regression Interpretation")
        lines.append("")
        lines.append(
            "Traditional logistic regression remains valuable here because its coefficients give a direct, readable risk-direction summary. "
            "In this dataset, variables such as academic pressure and financial stress tend to carry positive coefficients, while some lifestyle or demographic levels carry negative coefficients. "
            "These are predictive associations in a synthetic dataset, not causal explanations."
        )
        lines.append("")
        if not pos.empty:
            lines.append("Top positive logistic signals:")
            lines.append("")
            pos_table = pos[["feature_label", "level_label", "coefficient"]].copy()
            pos_table["level_label"] = pos_table["level_label"].fillna("numeric increase")
            pos_table["coefficient"] = pos_table["coefficient"].map(lambda value: f"{value:.3f}")
            lines.append(pos_table.rename(columns={"feature_label": "Feature", "level_label": "Level", "coefficient": "Coefficient"}).to_markdown(index=False))
            lines.append("")
        if not neg.empty:
            lines.append("Top negative logistic signals:")
            lines.append("")
            neg_table = neg[["feature_label", "level_label", "coefficient"]].copy()
            neg_table["level_label"] = neg_table["level_label"].fillna("numeric increase")
            neg_table["coefficient"] = neg_table["coefficient"].map(lambda value: f"{value:.3f}")
            lines.append(neg_table.rename(columns={"feature_label": "Feature", "level_label": "Level", "coefficient": "Coefficient"}).to_markdown(index=False))
            lines.append("")

    consensus = _feature_consensus(interpretation_outputs)
    if not consensus.empty:
        lines.append("### Cross-Model Consensus")
        lines.append("")
        lines.append(
            "A feature is more convincing as a dataset-level predictive signal when it appears repeatedly across model families."
        )
        lines.append("")
        lines.append(consensus.to_markdown(index=False))
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
