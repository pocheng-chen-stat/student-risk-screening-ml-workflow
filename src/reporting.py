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
    """Format cross-validation results for README display.

    Hyperparameter dictionaries are intentionally omitted from the README table
    because they are verbose and make GitHub Markdown hard to read. The full
    table, including selected hyperparameters, remains available in
    reports/cv_model_comparison.csv.
    """
    cv_columns = ["model", "best_cv_roc_auc", "cv_std_roc_auc"]
    available = [column for column in cv_columns if column in cv_comparison.columns]
    formatted = cv_comparison.loc[:, available].copy()
    if "model" in formatted.columns:
        formatted["model"] = formatted["model"].map(model_label)
    for column in ["best_cv_roc_auc", "cv_std_roc_auc"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
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


def _top_odds_ratios(table: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    data = table.sort_values("abs_coefficient", ascending=False).head(top_n).copy()
    data["Coefficient"] = data["coefficient"].map(lambda value: f"{value:.3f}")
    data["Odds Ratio exp(beta)"] = data["odds_ratio"].map(lambda value: f"{value:.2f}x")
    data["Approx. odds change"] = data["odds_change_percent"].map(lambda value: f"{value:+.0f}%")
    return data[
        ["feature_label", "comparison", "direction", "Coefficient", "Odds Ratio exp(beta)", "Approx. odds change"]
    ].rename(
        columns={
            "feature_label": "Feature",
            "comparison": "Comparison",
            "direction": "Direction",
        }
    )


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
    lines.append("## Results: Model Comparison")
    lines.append("")
    lines.append(
        f"The best held-out ROC-AUC is from **{model_label(best_auc['model'])}** "
        f"(ROC-AUC = {best_auc['roc_auc']:.3f}). "
        f"The highest sensitivity is from **{model_label(best_sensitivity['model'])}** "
        f"(sensitivity = {best_sensitivity['sensitivity_recall']:.3f})."
    )
    lines.append("")
    lines.append(
        "The performance gap among XGBoost, logistic regression, ridge logistic, lasso logistic, and linear SVM is small. "
        "Therefore, the project does not stop at selecting the highest-scoring model; it also uses interpretable models to explain which variables repeatedly carry predictive signal."
    )
    lines.append("")
    lines.append("### Held-Out Test Performance")
    lines.append("")
    lines.append(_format_metric_table(test_comparison).to_markdown(index=False, disable_numparse=True))
    lines.append("")
    lines.append("![Model comparison overview](reports/figures/model_comparison_overview.png)")
    lines.append("")
    lines.append("The overview figure combines ROC curves, precision-recall curves, and selected summary metrics. The individual ROC, precision-recall, and metric-summary figures are also saved under `reports/figures/` for detailed inspection.")
    lines.append("")
    lines.append("### How the Evaluation Metrics Are Read")
    lines.append("")
    lines.append("| Metric | Meaning in this project |")
    lines.append("|---|---|")
    lines.append("| Accuracy | Overall proportion of correct predictions. Useful as a general summary, but not enough by itself. |")
    lines.append("| ROC-AUC | Ability to rank risk-label cases above non-risk-label cases across thresholds. This is the main threshold-independent comparison metric. |")
    lines.append("| PR-AUC | Precision-recall performance, useful when the positive class is especially important. |")
    lines.append("| Sensitivity / Recall | Among students with the risk label, the proportion correctly detected. Important in a screening-style setting because missed high-risk cases are concerning. |")
    lines.append("| Specificity | Among students without the risk label, the proportion correctly identified as non-risk. |")
    lines.append("| Precision | Among students predicted as risk-label cases, the proportion that truly have the risk label. |")
    lines.append("| F1 Score | Harmonic mean of precision and recall. Useful when balancing false positives and false negatives. |")
    lines.append("")
    lines.append("### Overall Model Assessment")
    lines.append("")
    lines.append("The held-out results show that XGBoost has the strongest ROC-AUC, while KNN gives the highest sensitivity. However, logistic-based models remain very competitive. This means the project should not only choose the highest-scoring model; it should also use interpretable models to explain which variables are associated with the risk label.")
    lines.append("")
    lines.append("### Training-Set Cross-Validation")
    lines.append("")
    lines.append(
        "Cross-validation is performed only on the training set. The held-out test set is used once at the end to estimate final performance."
    )
    lines.append("")
    lines.append(_format_cv_table(cv_comparison).to_markdown(index=False, disable_numparse=True))
    lines.append("")
    lines.append("Selected hyperparameters are saved in `reports/cv_model_comparison.csv`. They are omitted from the README table to keep the model-comparison section readable.")
    lines.append("")
    lines.append("## Results: Feature Interpretation")
    lines.append("")
    lines.append(
        "After confirming that several models perform similarly well, the next question is: which explanatory variables are consistently selected? "
        "The figure below compares Lasso, Random Forest, and XGBoost. Lasso is sorted by absolute coefficient size, and the sign annotation keeps the positive/negative direction."
    )
    lines.append("")
    lines.append("![Cross-model feature importance](reports/figures/cross_model_feature_importance.png)")
    lines.append("")

    consensus = _feature_consensus(interpretation_outputs)
    if not consensus.empty:
        lines.append("### Cross-Model Consensus")
        lines.append("")
        lines.append(
            "Variables that repeatedly appear across model families are treated as stronger dataset-level predictive signals. "
            "This is still predictive agreement, not causal evidence."
        )
        lines.append("")
        lines.append(consensus.to_markdown(index=False, disable_numparse=True))
        lines.append("")

    odds = interpretation_outputs.get("logistic_odds_ratios_raw_scale", pd.DataFrame())
    if not odds.empty:
        lines.append("### Logistic Regression Odds-Ratio View")
        lines.append("")
        lines.append(
            "To make the logistic regression interpretation readable, an auxiliary logistic model is fit with numeric variables kept on their original scale. "
            "For numeric variables, `exp(beta)` is interpreted as the multiplicative change in the odds of the risk label for a one-unit increase, holding other variables fixed. "
            "For categorical variables, `exp(beta)` compares the displayed level with the reference level created during one-hot encoding."
        )
        lines.append("")
        lines.append(_top_odds_ratios(odds).to_markdown(index=False, disable_numparse=True))
        lines.append("")

    lasso = interpretation_outputs.get("lasso_feature_importance", pd.DataFrame())
    if not lasso.empty:
        lines.append("### Lasso Direction Summary")
        lines.append("")
        lines.append(
            "Lasso is sorted by absolute coefficient size to show variable strength, while the direction column keeps whether the selected signal is associated with higher or lower predicted risk."
        )
        lines.append("")
        lines.append(_top_lasso_features(lasso).to_markdown(index=False, disable_numparse=True))
        lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
