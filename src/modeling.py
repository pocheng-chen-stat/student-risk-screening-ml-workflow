"""Modeling utilities for tabular binary classification."""

from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


TARGET_COLUMN = "depression"


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def get_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into features and target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def make_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.25,
    random_state: int = 42,
) -> SplitData:
    """Create a stratified train/test split."""
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def identify_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical feature columns."""
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def _one_hot_encoder() -> OneHotEncoder:
    """Create an sklearn-version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing for a model family.

    Linear, distance-based, and margin-based models use scaled numeric features.
    Tree-based models keep numeric features unscaled but still require categorical
    encoding in sklearn workflows.
    """
    numeric_features, categorical_features = identify_feature_types(X)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def _linear_svc(random_state: int) -> LinearSVC:
    """Create a LinearSVC with version-compatible options."""
    try:
        return LinearSVC(max_iter=5000, random_state=random_state, dual="auto")
    except TypeError:
        return LinearSVC(max_iter=5000, random_state=random_state)


def build_model_specs(X: pd.DataFrame, *, random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    """Create models and compact tuning grids.

    The grids are intentionally compact so the project can run on a normal
    laptop. The goal is to demonstrate a correct workflow, not to perform an
    exhaustive benchmark.
    """
    scaled_preprocessor = build_preprocessor(X, scale_numeric=True)
    tree_preprocessor = build_preprocessor(X, scale_numeric=False)

    specs: dict[str, tuple[Pipeline, dict]] = {
        "logistic": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", LogisticRegression(max_iter=1000, solver="liblinear", penalty="l2")),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "lasso_logistic": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", LogisticRegression(max_iter=2000, penalty="l1", solver="liblinear")),
                ]
            ),
            {"model__C": [0.1, 1.0]},
        ),
        "ridge_logistic": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", LogisticRegression(max_iter=1000, solver="liblinear", penalty="l2")),
                ]
            ),
            {"model__C": [0.1, 1.0]},
        ),
        "knn": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", KNeighborsClassifier()),
                ]
            ),
            {"model__n_neighbors": [15, 25]},
        ),
        "svm_linear": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", _linear_svc(random_state)),
                ]
            ),
            {"model__C": [0.1, 1.0]},
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("preprocess", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            random_state=random_state,
                            n_jobs=1,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [120],
                "model__max_depth": [None, 10],
                "model__min_samples_leaf": [5],
            },
        ),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        specs["xgboost"] = (
            Pipeline(
                steps=[
                    ("preprocess", tree_preprocessor),
                    (
                        "model",
                        XGBClassifier(
                            objective="binary:logistic",
                            eval_metric="logloss",
                            random_state=random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__n_estimators": [100],
                "model__max_depth": [3],
                "model__learning_rate": [0.05],
            },
        )
    except Exception:
        pass

    return specs


def get_model_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return probability-like scores for ranking metrics."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Compute evaluation metrics for a binary classifier."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def tune_and_evaluate_models(
    split: SplitData,
    *,
    random_state: int = 42,
    cv_splits: int = 3,
    scoring: str = "roc_auc",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune models on the training set and evaluate once on held-out test data."""
    specs = build_model_specs(split.X_train, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    cv_rows = []
    test_rows = []
    results = {}

    for name, (pipeline, param_grid) in specs.items():
        if verbose:
            print(f"[modeling] Tuning {name} ...", flush=True)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
        )
        search.fit(split.X_train, split.y_train)

        best_model = search.best_estimator_
        y_pred = best_model.predict(split.X_test)
        y_score = get_model_score(best_model, split.X_test)
        metrics = compute_metrics(split.y_test, y_pred, y_score)

        cv_rows.append(
            {
                "model": name,
                "best_cv_roc_auc": search.best_score_,
                "cv_std_roc_auc": search.cv_results_["std_test_score"][search.best_index_],
                "best_params": str(search.best_params_),
            }
        )
        test_rows.append({"model": name, **metrics})
        results[name] = {
            "model": best_model,
            "best_params": search.best_params_,
            "y_test": split.y_test,
            "y_pred": y_pred,
            "y_score": y_score,
            "metrics": metrics,
        }

    cv_comparison = pd.DataFrame(cv_rows).sort_values("best_cv_roc_auc", ascending=False).reset_index(drop=True)
    test_comparison = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return cv_comparison, test_comparison, results
