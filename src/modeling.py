"""Modeling utilities for tabular binary classification."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
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
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def make_train_test_split(df: pd.DataFrame, *, test_size: float = 0.25, random_state: int = 42) -> SplitData:
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
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]
    return numeric_features, categorical_features


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, *, scale_numeric: bool) -> ColumnTransformer:
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


def make_logistic_regression(*, solver: str, l1_ratio: float, C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """Create LogisticRegression compatible with old and new sklearn versions.

    sklearn 1.8 deprecates the ``penalty`` argument in favor of ``l1_ratio``.
    Older versions still need ``penalty`` to request L1 regularization.
    """
    penalty_default = inspect.signature(LogisticRegression).parameters["penalty"].default
    if penalty_default == "deprecated":
        return LogisticRegression(max_iter=max_iter, solver=solver, C=C, l1_ratio=l1_ratio)

    penalty = "l1" if l1_ratio == 1.0 else "l2"
    return LogisticRegression(max_iter=max_iter, solver=solver, C=C, penalty=penalty)


def build_model_specs(X: pd.DataFrame, *, random_state: int = 42) -> dict[str, tuple[Pipeline, dict]]:
    scaled_preprocessor = build_preprocessor(X, scale_numeric=True)
    tree_preprocessor = build_preprocessor(X, scale_numeric=False)

    specs: dict[str, tuple[Pipeline, dict]] = {
        "logistic": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", make_logistic_regression(max_iter=1000, solver="liblinear", l1_ratio=0.0)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "lasso_logistic": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", make_logistic_regression(max_iter=1000, solver="liblinear", l1_ratio=1.0)),
                ]
            ),
            {"model__C": [0.05, 0.1, 1.0]},
        ),
        "ridge_logistic": (
            Pipeline(
                steps=[
                    ("preprocess", scaled_preprocessor),
                    ("model", make_logistic_regression(max_iter=1000, solver="liblinear", l1_ratio=0.0)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
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
                    ("model", LinearSVC(max_iter=5000, random_state=random_state)),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
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
                "model__n_estimators": [150],
                "model__max_depth": [None, 8],
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
                "model__n_estimators": [150],
                "model__max_depth": [3],
                "model__learning_rate": [0.05],
            },
        )
    except Exception:
        pass

    return specs


def get_model_score(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
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
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    specs = build_model_specs(split.X_train, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    cv_rows = []
    test_rows = []
    results = {}

    for name, (pipeline, param_grid) in specs.items():
        print(f"[modeling] Tuning {name} ...")
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

        print(f"[modeling] Finished {name}: CV ROC-AUC={search.best_score_:.3f}, test ROC-AUC={metrics['roc_auc']:.3f}")

        cv_rows.append(
            {
                "model": name,
                "best_cv_roc_auc": search.best_score_,
                "cv_std_roc_auc": search.cv_results_["std_test_score"][search.best_index_],
                "best_params": search.best_params_,
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
