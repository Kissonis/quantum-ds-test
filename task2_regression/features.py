"""Feature engineering shared by train.py and predict.py.

The exploratory analysis (``notebooks/eda.ipynb``) shows that the target is a
deterministic function of two features::

    target = feature_6 ** 2 + feature_7

All other 51 columns are noise.  Keeping the transformation in one place
guarantees that training and inference always see identical inputs.
"""

from __future__ import annotations

import pandas as pd

#: Columns the target depends on (as named in the CSV files).
QUADRATIC_FEATURE = "6"
LINEAR_FEATURE = "7"

#: Names of the engineered columns fed to the linear model.
ENGINEERED_COLUMNS = ["f6_squared", "f7"]


def make_linear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the two engineered features for the linear model.

    Parameters
    ----------
    df:
        Raw frame with at least columns ``"6"`` and ``"7"``.

    Returns
    -------
    pd.DataFrame
        Frame with columns ``f6_squared`` and ``f7``.
    """
    missing = [c for c in (QUADRATIC_FEATURE, LINEAR_FEATURE) if c not in df.columns]
    if missing:
        raise KeyError(f"input is missing required columns: {missing}")
    return pd.DataFrame(
        {
            "f6_squared": df[QUADRATIC_FEATURE].astype(float) ** 2,
            "f7": df[LINEAR_FEATURE].astype(float),
        },
        index=df.index,
    )


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    """All feature columns of a raw frame (everything except ``target``)."""
    return [c for c in df.columns if c != "target"]
