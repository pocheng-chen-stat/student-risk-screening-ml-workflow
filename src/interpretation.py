"""Model interpretation utilities.

Interpretation is limited to predictive signals and correlates. This project
is not a clinical, psychological, or causal study.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.feature_display import (
    add_feature_labels,
    aggregate_coefficients_by_feature,
    aggregate_importance_by_feature,
    parse_pipeline_feature,
)


def _one_hot_encoder_drop_first() -> OneHotEncoder:
    """Return OneHotEncoder with a reference category for odds-ratio interpretation."""
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def get_feature_names(fitted_pipeline) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out().tolist()


def model_coefficients(fitted_pipeline) -> pd.DataFrame:
    names = get_feature_names(fitted_pipeline)
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient", "abs_coefficient"])
    coefs = model.coef_.ravel()
    return (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def tree_feature_importance(fitted_pipeline) -> pd.DataFrame:
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
    n_repeats: int = 5,
) -> pd.DataFrame:
    result = permutation_importance(
        fitted_pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=1,
    )
    table = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return add_feature_labels(table)


def logistic_odds_ratio_table(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Fit an interpretable logistic model and return odds ratios.

    The comparison models standardize numeric variables for predictive modeling.
    For interpretation, this auxiliary logistic model deliberately keeps numeric
    variables on their original scale and uses one-hot encoding with a reference
    category. Therefore numeric coefficients can be read as a one-raw-unit
    increase, while categorical coefficients are compared with a reference level.
    """
    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X_train.columns if column not in numeric_features]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder_drop_first()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = LogisticRegression(max_iter=2000, solver="liblinear", C=1_000_000.0)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    coefficients = pipeline.named_steps["model"].coef_.ravel()

    rows = []
    for raw_feature, coefficient in zip(feature_names, coefficients):
        parsed = parse_pipeline_feature(raw_feature, known_features=list(X_train.columns))
        odds_ratio = float(np.exp(coefficient))
        if parsed.category_level is None:
            comparison = "one-unit increase"
        else:
            comparison = f"{parsed.category_label} vs. reference level"
        rows.append(
            {
                "feature": parsed.base_feature,
                "feature_label": parsed.base_label,
                "level": parsed.category_level,
                "level_label": parsed.category_label,
                "comparison": comparison,
                "coefficient": float(coefficient),
                "abs_coefficient": abs(float(coefficient)),
                "odds_ratio": odds_ratio,
                "odds_change_percent": (odds_ratio - 1.0) * 100.0,
                "direction": "Higher odds" if odds_ratio >= 1 else "Lower odds",
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def create_interpretation_outputs(
    model_results: dict,
    X_test,
    y_test,
    X_train=None,
    y_train=None,
) -> dict[str, pd.DataFrame]:
    """Create raw and aggregated interpretation tables.

    Raw sklearn features are kept for transparency, while aggregated tables are
    used for readable README figures. Aggregation maps one-hot encoded levels
    back to their original explanatory variables.
    """
    outputs: dict[str, pd.DataFrame] = {}
    original_features = list(X_test.columns)

    if "logistic" in model_results:
        logistic_raw = model_coefficients(model_results["logistic"]["model"])
        outputs["logistic_coefficients"] = logistic_raw
        outputs["logistic_risk_direction"] = aggregate_coefficients_by_feature(
            logistic_raw,
            known_features=original_features,
        )

    if X_train is not None and y_train is not None:
        outputs["logistic_odds_ratios_raw_scale"] = logistic_odds_ratio_table(X_train, y_train)

    if "lasso_logistic" in model_results:
        lasso_raw = model_coefficients(model_results["lasso_logistic"]["model"])
        outputs["lasso_coefficients"] = lasso_raw
        outputs["lasso_feature_importance"] = aggregate_coefficients_by_feature(
            lasso_raw,
            known_features=original_features,
        )

    for model_name in ["random_forest", "xgboost"]:
        if model_name in model_results:
            table = tree_feature_importance(model_results[model_name]["model"])
            if not table.empty:
                outputs[f"{model_name}_importance"] = table
                outputs[f"{model_name}_feature_importance"] = aggregate_importance_by_feature(
                    table,
                    known_features=original_features,
                    value_col="importance",
                )

    best_model_name = max(model_results, key=lambda name: model_results[name]["metrics"]["roc_auc"])
    outputs["permutation_importance"] = permutation_importance_table(
        model_results[best_model_name]["model"], X_test, y_test
    )

    return outputs
