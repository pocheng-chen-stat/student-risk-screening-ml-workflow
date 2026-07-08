"""Run the student risk screening ML workflow.

The project is for machine learning workflow demonstration only. It is not a
clinical diagnostic tool and does not make causal claims.
"""

from __future__ import annotations

from pathlib import Path

from src.cleaning import clean_dataset, save_cleaning_report
from src.data_loader import load_raw_data
from src.eda import depression_rate_by_feature, save_eda_tables
from src.interpretation import create_interpretation_outputs
from src.modeling import make_train_test_split, tune_and_evaluate_models
from src.reporting import save_readme_results
from src.visualization import (
    ensure_dir,
    plot_confusion_matrix,
    plot_cross_model_feature_importance,
    plot_depression_rate_by_feature,
    plot_feature_bar,
    plot_missing_values,
    plot_model_metric_comparison,
    plot_numeric_correlation,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_target_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_depression_dataset.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_depression_cleaned.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    ensure_dir(REPORTS_DIR)
    ensure_dir(FIGURES_DIR)
    ensure_dir(PROCESSED_DATA_PATH.parent)

    raw_df = load_raw_data(RAW_DATA_PATH)
    cleaned_df, cleaning_report = clean_dataset(raw_df)
    cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
    save_cleaning_report(cleaning_report, REPORTS_DIR / "cleaning_report.csv")

    save_eda_tables(cleaned_df, REPORTS_DIR)
    plot_target_distribution(cleaned_df, FIGURES_DIR / "target_distribution.png")
    plot_missing_values(cleaned_df, FIGURES_DIR / "missing_values.png")
    plot_numeric_correlation(cleaned_df, FIGURES_DIR / "numeric_correlation.png")

    for feature in [
        "academic_pressure",
        "financial_stress",
        "sleep_duration",
        "dietary_habits",
        "have_you_ever_had_suicidal_thoughts",
        "family_history_of_mental_illness",
        "degree",
    ]:
        if feature in cleaned_df.columns:
            depression_rate_by_feature(cleaned_df, feature).to_csv(
                REPORTS_DIR / f"depression_rate_by_{feature}.csv", index=False
            )
            plot_depression_rate_by_feature(
                cleaned_df,
                feature,
                FIGURES_DIR / f"depression_rate_by_{feature}.png",
            )

    split = make_train_test_split(cleaned_df)
    cv_comparison, test_comparison, model_results = tune_and_evaluate_models(split)
    cv_comparison.to_csv(REPORTS_DIR / "cv_model_comparison.csv", index=False)
    test_comparison.to_csv(REPORTS_DIR / "test_model_comparison.csv", index=False)

    plot_model_metric_comparison(test_comparison, FIGURES_DIR / "model_metric_comparison.png")
    plot_roc_curves(model_results, FIGURES_DIR / "roc_curve_comparison.png")
    plot_precision_recall_curves(model_results, FIGURES_DIR / "precision_recall_curve_comparison.png")

    best_auc_model = test_comparison.iloc[0]["model"]
    highest_sensitivity_model = test_comparison.sort_values("sensitivity_recall", ascending=False).iloc[0]["model"]

    plot_confusion_matrix(
        model_results[best_auc_model]["y_test"],
        model_results[best_auc_model]["y_pred"],
        f"Confusion Matrix: Best ROC-AUC ({best_auc_model})",
        FIGURES_DIR / "confusion_matrix_best_auc.png",
    )
    plot_confusion_matrix(
        model_results[highest_sensitivity_model]["y_test"],
        model_results[highest_sensitivity_model]["y_pred"],
        f"Confusion Matrix: Highest Sensitivity ({highest_sensitivity_model})",
        FIGURES_DIR / "confusion_matrix_highest_sensitivity.png",
    )

    interpretation_outputs = create_interpretation_outputs(model_results, split.X_test, split.y_test, split.X_train, split.y_train)
    for name, table in interpretation_outputs.items():
        table.to_csv(REPORTS_DIR / f"{name}.csv", index=False)

    plot_specs = {
        "lasso_feature_importance": ("feature_label", "abs_coefficient", "Top Lasso Features by Absolute Coefficient"),
        "random_forest_feature_importance": ("feature_label", "importance", "Top Random Forest Feature Importances"),
        "xgboost_feature_importance": ("feature_label", "importance", "Top XGBoost Feature Importances"),
        "permutation_importance": ("feature_label", "importance_mean", "Top Permutation Importances"),
    }
    for name, (feature_col, value_col, title) in plot_specs.items():
        if name in interpretation_outputs:
            plot_feature_bar(
                interpretation_outputs[name],
                feature_col,
                value_col,
                title,
                FIGURES_DIR / f"top_{name}.png",
            )

    plot_cross_model_feature_importance(
        lasso_table=interpretation_outputs.get("lasso_feature_importance"),
        random_forest_table=interpretation_outputs.get("random_forest_feature_importance"),
        xgboost_table=interpretation_outputs.get("xgboost_feature_importance"),
        output_path=FIGURES_DIR / "cross_model_feature_importance.png",
    )

    save_readme_results(
        test_comparison=test_comparison,
        cv_comparison=cv_comparison,
        interpretation_outputs=interpretation_outputs,
        output_path=REPORTS_DIR / "readme_results.md",
    )

    print("Workflow completed.")
    print(f"Reports saved to: {REPORTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("\nCross-validation comparison:")
    print(cv_comparison.to_string(index=False))
    print("\nHeld-out test comparison:")
    print(test_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
