#!/usr/bin/env python
"""Train a regression model on train.csv and save it to disk.

Two models are available:

``linear`` (default)
    Linear regression on two engineered features, ``feature_6 ** 2`` and
    ``feature_7``.  EDA shows the target is exactly this function, so the
    model reaches RMSE ~1e-14 and is the one used for the final predictions.

``lgbm``
    LightGBM gradient boosting on all 53 raw features.  Kept as a generic,
    assumption-free baseline for comparison; it is competitive but cannot
    represent the parabola exactly, so its RMSE is noticeably higher.

Both models are evaluated with K-fold cross-validation and on a hold-out
split before the final fit on the full training set.

Usage
-----
    python train.py                                  # linear model
    python train.py --model lgbm --output models/lgbm.joblib
    python train.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from features import ENGINEERED_COLUMNS, make_linear_features, raw_feature_columns

HERE = Path(__file__).resolve().parent
DEFAULT_TRAIN = HERE / "data" / "train.csv"
DEFAULT_OUTPUT = HERE / "models" / "model.joblib"

log = logging.getLogger("train")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_model(name: str, seed: int) -> Tuple[BaseEstimator, str]:
    """Return an unfitted estimator and the feature mode it expects.

    The feature mode (``"linear"`` or ``"raw"``) tells :func:`prepare_features`
    which columns to build; it is stored next to the model so that
    ``predict.py`` can reproduce the exact same inputs.
    """
    if name == "linear":
        return LinearRegression(), "linear"
    if name == "lgbm":
        import lightgbm as lgb  # imported lazily: optional dependency

        # No column subsampling: 51 of the 53 features are noise, so dropping
        # the two informative ones from a tree (as colsample_bytree would)
        # hurts far more than the regularisation helps.
        model = lgb.LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
        )
        return model, "raw"
    raise ValueError(f"unknown model {name!r}; choose 'linear' or 'lgbm'")


def prepare_features(df: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    """Build the model inputs from a raw frame according to ``feature_mode``."""
    if feature_mode == "linear":
        return make_linear_features(df)
    if feature_mode == "raw":
        return df[raw_feature_columns(df)]
    raise ValueError(f"unknown feature mode {feature_mode!r}")


def cross_validate(model: BaseEstimator, x: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> Dict[str, Any]:
    """K-fold CV; returns per-fold RMSE plus mean/std."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for i, (tr_idx, va_idx) in enumerate(kf.split(x), start=1):
        fold_model = clone(model)  # fresh unfitted copy each fold
        fold_model.fit(x.iloc[tr_idx], y.iloc[tr_idx])
        score = rmse(y.iloc[va_idx], fold_model.predict(x.iloc[va_idx]))
        scores.append(score)
        log.info("fold %d/%d  RMSE = %.6g", i, folds, score)
    return {"fold_rmse": scores, "mean_rmse": float(np.mean(scores)), "std_rmse": float(np.std(scores))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="path to train.csv")
    parser.add_argument("--model", choices=["linear", "lgbm"], default="linear", help="model to train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to save the fitted model")
    parser.add_argument("--folds", type=int, default=5, help="number of CV folds (0 to skip CV)")
    parser.add_argument("--holdout", type=float, default=0.2, help="hold-out fraction for a sanity check (0 to skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.train.exists():
        log.error("training file not found: %s (datasets are not in git, see data/README.md)", args.train)
        return 1

    t0 = time.time()
    df = pd.read_csv(args.train)
    if "target" not in df.columns:
        log.error("'target' column not found in %s", args.train)
        return 1
    log.info("loaded %s: %d rows, %d columns", args.train, *df.shape)

    model, feature_mode = build_model(args.model, args.seed)
    x = prepare_features(df, feature_mode)
    y = df["target"]
    log.info("model=%s  features=%s (%d columns)", args.model, feature_mode, x.shape[1])

    metrics: Dict[str, Any] = {"model": args.model, "feature_mode": feature_mode, "n_train": int(len(df))}

    if args.holdout > 0:
        x_tr, x_va, y_tr, y_va = train_test_split(x, y, test_size=args.holdout, random_state=args.seed)
        hold_model = clone(model)
        hold_model.fit(x_tr, y_tr)
        metrics["holdout_rmse"] = rmse(y_va, hold_model.predict(x_va))
        log.info("hold-out (%.0f%%) RMSE = %.6g", args.holdout * 100, metrics["holdout_rmse"])

    if args.folds > 1:
        metrics["cv"] = cross_validate(model, x, y, args.folds, args.seed)
        log.info("%d-fold CV RMSE = %.6g +/- %.3g", args.folds, metrics["cv"]["mean_rmse"], metrics["cv"]["std_rmse"])

    # Final fit on the full training set.
    model.fit(x, y)
    metrics["train_rmse"] = rmse(y, model.predict(x))
    log.info("train RMSE (full fit) = %.6g", metrics["train_rmse"])
    if feature_mode == "linear":
        coefs = dict(zip(ENGINEERED_COLUMNS, model.coef_.round(6).tolist()))
        log.info("coefficients: %s  intercept: %.6g", coefs, model.intercept_)
        metrics["coefficients"] = coefs
        metrics["intercept"] = float(model.intercept_)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_mode": feature_mode, "model_name": args.model, "metrics": metrics}
    joblib.dump(bundle, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("saved model to %s and metrics to %s (%.1fs)", args.output, metrics_path, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
