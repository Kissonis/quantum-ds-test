"""Tests for the feature engineering and the train -> predict pipeline.

They use a small synthetic frame with the same layout as ``train.csv``
(columns ``"0"`` .. ``"52"`` plus ``target``), so they run in seconds and do
not need the real dataset.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import predict  # noqa: E402
import train  # noqa: E402
from features import ENGINEERED_COLUMNS, make_linear_features, raw_feature_columns  # noqa: E402

N_FEATURES = 53


def make_frame(n_rows: int = 400, seed: int = 0, with_target: bool = True) -> pd.DataFrame:
    """Synthetic data that follows ``target = f6**2 + f7`` like the real set."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.integers(0, 500, size=(n_rows, N_FEATURES)).astype(float),
        columns=[str(i) for i in range(N_FEATURES)],
    )
    df["6"] = rng.choice([-1, 1], n_rows) * np.sqrt(rng.integers(0, 100, n_rows))
    df["7"] = rng.random(n_rows)
    if with_target:
        df["target"] = df["6"] ** 2 + df["7"]
    return df


# --- features ---------------------------------------------------------------

def test_make_linear_features_values():
    df = pd.DataFrame({"6": [-3.0, 2.0], "7": [0.5, 0.25], "8": [1, 2]})
    out = make_linear_features(df)
    assert list(out.columns) == ENGINEERED_COLUMNS
    np.testing.assert_allclose(out["f6_squared"], [9.0, 4.0])
    np.testing.assert_allclose(out["f7"], [0.5, 0.25])
    assert out.index.equals(df.index)


def test_make_linear_features_missing_column():
    with pytest.raises(KeyError, match="missing required columns"):
        make_linear_features(pd.DataFrame({"6": [1.0]}))


def test_raw_feature_columns_excludes_target():
    df = make_frame(5)
    cols = raw_feature_columns(df)
    assert "target" not in cols
    assert len(cols) == N_FEATURES


# --- end-to-end pipeline ------------------------------------------------------

def test_linear_pipeline_recovers_formula(tmp_path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    out_csv = tmp_path / "pred.csv"

    make_frame(400, seed=1).to_csv(train_csv, index=False)
    test_df = make_frame(100, seed=2, with_target=False)
    test_df.to_csv(test_csv, index=False)

    assert train.main(["--train", str(train_csv), "--output", str(model_path), "--folds", "3"]) == 0
    bundle = joblib.load(model_path)
    assert bundle["feature_mode"] == "linear"
    np.testing.assert_allclose(bundle["model"].coef_, [1.0, 1.0], atol=1e-8)
    assert bundle["metrics"]["cv"]["mean_rmse"] < 1e-8
    assert model_path.with_suffix(".metrics.json").exists()

    assert predict.main(["--model", str(model_path), "--input", str(test_csv), "--output", str(out_csv)]) == 0
    preds = pd.read_csv(out_csv)
    assert list(preds.columns) == ["target"]
    assert len(preds) == len(test_df)
    np.testing.assert_allclose(preds["target"], test_df["6"] ** 2 + test_df["7"], atol=1e-8)


def test_lgbm_pipeline_runs(tmp_path):
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "lgbm.joblib"
    make_frame(300, seed=3).to_csv(train_csv, index=False)

    code = train.main([
        "--train", str(train_csv), "--model", "lgbm", "--output", str(model_path),
        "--folds", "0", "--holdout", "0",
    ])
    assert code == 0
    bundle = joblib.load(model_path)
    assert bundle["feature_mode"] == "raw"
    preds = predict.predict_frame(bundle, make_frame(20, seed=4, with_target=False))
    assert preds.shape == (20,)
    assert np.isfinite(preds).all()


def test_train_reports_missing_file(tmp_path):
    assert train.main(["--train", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "m.joblib")]) == 1


def test_predict_reports_missing_model(tmp_path):
    assert predict.main(["--model", str(tmp_path / "nope.joblib")]) == 1
