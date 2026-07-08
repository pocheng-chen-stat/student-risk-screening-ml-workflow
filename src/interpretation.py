"""Model interpretation utilities.

Interpretation is limited to predictive signals and correlates. This project
does not make causal, medical, or clinical claims.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_names(fitted_pipeline) -> list[str]:
    """Return transformed feature names from a fitted pipeline."""
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out().tolist()


def lasso_coefficients(fitted_pipeline) -> pd.DataFrame:
    """Extract Lasso logistic regression coefficients on transformed features."""
    names = get_feature_names(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]
    coefs = model.coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def tree_feature_importance(fitted_pipeline) -> pd.DataFrame:
    """Extract feature importances from a fitted tree-based model."""
    names = get_feature_names(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    return (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def permutation_importance_table(
    fitted_pipeline,
    X_test,
    y_test,
    *,
    random_state: int = 42,
    n_repeats: int = 3,
    max_rows: int = 1500,
) -> pd.DataFrame:
    """Compute permutation importance on a reproducible subset of the test set.

    Permutation importance can be slow on a full test set. A fixed subset keeps
    the portfolio workflow fast while still providing an interpretable ranking.
    """
    if len(X_test) > max_rows:
        X_eval = X_test.sample(n=max_rows, random_state=random_state)
        y_eval = y_test.loc[X_eval.index]
    else:
        X_eval = X_test
        y_eval = y_test

    result = permutation_importance(
        fitted_pipeline,
        X_eval,
        y_eval,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X_eval.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def create_interpretation_outputs(model_results: dict, X_test, y_test) -> dict[str, pd.DataFrame]:
    """Create model interpretation tables."""
    outputs: dict[str, pd.DataFrame] = {}

    if "lasso_logistic" in model_results:
        outputs["lasso_coefficients"] = lasso_coefficients(model_results["lasso_logistic"]["model"])

    for model_name in ["random_forest", "xgboost"]:
        if model_name in model_results:
            table = tree_feature_importance(model_results[model_name]["model"])
            if not table.empty:
                outputs[f"{model_name}_importance"] = table

    best_model_name = max(model_results, key=lambda name: model_results[name]["metrics"]["roc_auc"])
    outputs["permutation_importance"] = permutation_importance_table(
        model_results[best_model_name]["model"], X_test, y_test
    )
    return outputs
