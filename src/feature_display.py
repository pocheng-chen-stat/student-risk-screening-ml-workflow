"""Helpers for human-readable feature labels and feature-importance aggregation.

The modeling pipeline keeps machine-friendly column names such as
``academic_pressure``.  GitHub figures and tables should instead use labels such
as ``Academic Pressure`` so the project reads like an analysis report rather
than raw code output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


LABEL_OVERRIDES = {
    "cgpa": "CGPA",
    "have_you_ever_had_suicidal_thoughts": "Suicidal Thoughts",
    "family_history_of_mental_illness": "Family History of Mental Illness",
    "work_study_hours": "Work/Study Hours",
    "work_pressure": "Work Pressure",
    "job_satisfaction": "Job Satisfaction",
}


@dataclass(frozen=True)
class ParsedFeature:
    raw_feature: str
    base_feature: str
    category_level: str | None
    base_label: str
    category_label: str | None


def display_feature_label(feature: str) -> str:
    """Convert a snake_case feature name into a readable label."""
    feature = str(feature).strip().strip("'").strip('"')
    return LABEL_OVERRIDES.get(feature, feature.replace("_", " ").title())


def display_category_label(category: str | None) -> str | None:
    if category is None:
        return None
    cleaned = str(category).strip().strip("'").strip('"')
    return cleaned.replace("_", " ").title()


def _strip_pipeline_prefix(raw_feature: str) -> str:
    feature = str(raw_feature)
    for prefix in ["numeric__", "categorical__", "remainder__"]:
        if feature.startswith(prefix):
            return feature[len(prefix) :]
    return feature


def parse_pipeline_feature(raw_feature: str, known_features: Iterable[str] | None = None) -> ParsedFeature:
    """Parse a transformed sklearn feature name back to its original feature.

    Examples
    --------
    ``numeric__academic_pressure`` -> Academic Pressure
    ``categorical__dietary_habits_Unhealthy`` -> Dietary Habits, Unhealthy
    """
    stripped = _strip_pipeline_prefix(raw_feature)
    known = sorted([str(feature) for feature in (known_features or [])], key=len, reverse=True)

    for base in known:
        if stripped == base:
            return ParsedFeature(
                raw_feature=raw_feature,
                base_feature=base,
                category_level=None,
                base_label=display_feature_label(base),
                category_label=None,
            )
        prefix = f"{base}_"
        if stripped.startswith(prefix):
            category = stripped[len(prefix) :]
            return ParsedFeature(
                raw_feature=raw_feature,
                base_feature=base,
                category_level=category,
                base_label=display_feature_label(base),
                category_label=display_category_label(category),
            )

    return ParsedFeature(
        raw_feature=raw_feature,
        base_feature=stripped,
        category_level=None,
        base_label=display_feature_label(stripped),
        category_label=None,
    )


def aggregate_coefficients_by_feature(
    coefficient_table: pd.DataFrame,
    *,
    known_features: Iterable[str],
    coefficient_col: str = "coefficient",
) -> pd.DataFrame:
    """Aggregate coefficient table to one row per original explanatory variable.

    For one-hot encoded categorical variables, the level with the largest
    absolute coefficient is used to summarize the variable direction.  This
    keeps the plot readable while still retaining the positive/negative signal.
    """
    if coefficient_table.empty or coefficient_col not in coefficient_table.columns:
        return pd.DataFrame()

    rows = []
    for _, row in coefficient_table.iterrows():
        parsed = parse_pipeline_feature(row["feature"], known_features)
        coefficient = float(row[coefficient_col])
        rows.append(
            {
                "feature": parsed.base_feature,
                "feature_label": parsed.base_label,
                "level": parsed.category_level,
                "level_label": parsed.category_label,
                "coefficient": coefficient,
                "abs_coefficient": abs(coefficient),
                "direction": "Positive" if coefficient >= 0 else "Negative",
            }
        )

    expanded = pd.DataFrame(rows)
    idx = expanded.groupby("feature")["abs_coefficient"].idxmax()
    return expanded.loc[idx].sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def aggregate_importance_by_feature(
    importance_table: pd.DataFrame,
    *,
    known_features: Iterable[str],
    value_col: str = "importance",
) -> pd.DataFrame:
    """Aggregate one-hot importance values back to original variables."""
    if importance_table.empty or value_col not in importance_table.columns:
        return pd.DataFrame()

    rows = []
    for _, row in importance_table.iterrows():
        parsed = parse_pipeline_feature(row["feature"], known_features)
        rows.append(
            {
                "feature": parsed.base_feature,
                "feature_label": parsed.base_label,
                value_col: float(row[value_col]),
            }
        )

    return (
        pd.DataFrame(rows)
        .groupby(["feature", "feature_label"], as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .reset_index(drop=True)
    )


def add_feature_labels(table: pd.DataFrame, feature_col: str = "feature") -> pd.DataFrame:
    """Add readable labels to a table that already has original feature names."""
    if table.empty or feature_col not in table.columns:
        return table.copy()
    output = table.copy()
    output["feature_label"] = output[feature_col].map(display_feature_label)
    return output
