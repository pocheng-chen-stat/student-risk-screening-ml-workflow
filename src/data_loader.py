"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {path}. "
            "Place student_depression_dataset.csv under data/raw/ before running."
        )
    return pd.read_csv(path)
