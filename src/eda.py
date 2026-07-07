"""Exploratory data analysis tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def target_summary(df: pd.DataFrame, target: str = "depression") -> pd.DataFrame:
    counts = df[target].value_counts().sort_index()
    rates = df[target].value_counts(normalize=True).sort_index()
    return pd.DataFrame({"class": counts.index, "count": counts.values, "rate": rates.values})


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "feature"})
        .assign(missing_rate=lambda d: d["missing_count"] / len(df))
        .sort_values("missing_count", ascending=False)
    )


def depression_rate_by_feature(df: pd.DataFrame, feature: str, target: str = "depression") -> pd.DataFrame:
    return (
        df.groupby(feature, dropna=False)[target]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "depression_rate"})
        .sort_values("depression_rate", ascending=False)
    )


def save_eda_tables(df: pd.DataFrame, reports_dir: str | Path) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    target_summary(df).to_csv(reports_dir / "target_summary.csv", index=False)
    missing_summary(df).to_csv(reports_dir / "missing_summary.csv", index=False)
