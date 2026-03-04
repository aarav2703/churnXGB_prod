"""
Load raw data for ChurnXGB.

This module is intentionally minimal: it loads the raw CSV without
transforming business logic. Cleaning happens in data/clean.py.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    """
    Load the Online Retail II raw CSV.

    Parameters
    ----------
    path : str | Path
        Path to raw CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataframe.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path.resolve()}")

    # Read as raw first; cleaning/parsing happens elsewhere
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df
