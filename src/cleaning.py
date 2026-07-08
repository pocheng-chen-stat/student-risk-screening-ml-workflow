"""Cleaning rules for the Student Depression Dataset case study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


TARGET_COLUMN = "depression"


@dataclass
class CleaningReport:
    raw_rows: int
    removed_non_student_rows: int
    removed_cgpa_zero_rows: int
    removed_degree_other_rows: int
    removed_financial_stress_invalid_rows: int
    final_rows: int
    final_positive_rate: float
    dropped_columns: str


def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: normalize_column_name(column) for column in df.columns})


def _count_non_student_rows(df: pd.DataFrame) -> int:
    if "profession" not in df.columns:
        return 0
    return int(df["profession"].astype(str).str.strip().str.lower().ne("student").sum())


def _count_cgpa_zero_rows(df: pd.DataFrame) -> int:
    if "cgpa" not in df.columns:
        return 0
    cgpa = pd.to_numeric(df["cgpa"], errors="coerce")
    return int(cgpa.eq(0).sum())


def _count_degree_other_rows(df: pd.DataFrame) -> int:
    if "degree" not in df.columns:
        return 0
    return int(df["degree"].astype(str).str.strip().str.lower().isin({"other", "others"}).sum())


def _count_invalid_financial_stress_rows(df: pd.DataFrame) -> int:
    if "financial_stress" not in df.columns:
        return 0
    numeric_values = pd.to_numeric(df["financial_stress"], errors="coerce")
    return int(numeric_values.isna().sum())


def _convert_numeric_if_possible(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError):
        return series


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    cleaned = normalize_columns(df).copy()
    raw_rows = len(cleaned)
    dropped_columns: list[str] = []

    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])
        dropped_columns.append("id")

    if "city" in cleaned.columns:
        cleaned = cleaned.drop(columns=["city"])
        dropped_columns.append("city")

    removed_non_student_rows = _count_non_student_rows(cleaned)
    if "profession" in cleaned.columns:
        is_student = cleaned["profession"].astype(str).str.strip().str.lower().eq("student")
        cleaned = cleaned.loc[is_student].copy()
        cleaned = cleaned.drop(columns=["profession"])
        dropped_columns.append("profession")

    removed_cgpa_zero_rows = _count_cgpa_zero_rows(cleaned)
    if "cgpa" in cleaned.columns:
        cleaned["cgpa"] = pd.to_numeric(cleaned["cgpa"], errors="coerce")
        cleaned = cleaned.loc[cleaned["cgpa"].ne(0)].copy()

    removed_degree_other_rows = _count_degree_other_rows(cleaned)
    if "degree" in cleaned.columns:
        is_other_degree = cleaned["degree"].astype(str).str.strip().str.lower().isin({"other", "others"})
        cleaned = cleaned.loc[~is_other_degree].copy()

    removed_financial_stress_invalid_rows = _count_invalid_financial_stress_rows(cleaned)
    if "financial_stress" in cleaned.columns:
        cleaned["financial_stress"] = pd.to_numeric(cleaned["financial_stress"], errors="coerce")
        cleaned = cleaned.loc[cleaned["financial_stress"].notna()].copy()

    for column in cleaned.columns:
        if column == TARGET_COLUMN:
            continue
        cleaned[column] = _convert_numeric_if_possible(cleaned[column])

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Target column missing after cleaning: {TARGET_COLUMN}")

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    cleaned = cleaned.loc[cleaned[TARGET_COLUMN].isin([0, 1])].copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    cleaned = cleaned.replace({np.nan: None}).reset_index(drop=True)

    report = CleaningReport(
        raw_rows=raw_rows,
        removed_non_student_rows=removed_non_student_rows,
        removed_cgpa_zero_rows=removed_cgpa_zero_rows,
        removed_degree_other_rows=removed_degree_other_rows,
        removed_financial_stress_invalid_rows=removed_financial_stress_invalid_rows,
        final_rows=len(cleaned),
        final_positive_rate=float(cleaned[TARGET_COLUMN].mean()) if len(cleaned) else 0.0,
        dropped_columns=", ".join(dropped_columns),
    )
    return cleaned, report


def save_cleaning_report(report: CleaningReport, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(report)]).to_csv(output_path, index=False)
