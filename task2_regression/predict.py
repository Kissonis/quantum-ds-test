#!/usr/bin/env python
"""Run a saved model on a CSV file and write predictions.

The model bundle produced by ``train.py`` stores the feature mode next to the
estimator, so this script rebuilds exactly the inputs the model was trained on.

Usage
-----
    python predict.py                                   # data/hidden_test.csv -> predictions.csv
    python predict.py --input other.csv --output out.csv
    python predict.py --model models/lgbm.joblib
    python predict.py --help

If the input file contains a ``target`` column the RMSE is reported as well.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from train import prepare_features, rmse

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "models" / "model.joblib"
DEFAULT_INPUT = HERE / "data" / "hidden_test.csv"
DEFAULT_OUTPUT = HERE / "predictions.csv"

log = logging.getLogger("predict")


def predict_frame(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Return predictions for a raw frame using a bundle saved by train.py."""
    x = prepare_features(df, bundle["feature_mode"])
    return np.asarray(bundle["model"].predict(x), dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="model bundle saved by train.py")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV with the 53 feature columns")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to write predictions")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    if not args.model.exists():
        log.error("model not found: %s (run train.py first)", args.model)
        return 1
    if not args.input.exists():
        log.error("input file not found: %s (datasets are not in git, see data/README.md)", args.input)
        return 1

    bundle = joblib.load(args.model)
    log.info("loaded %s model from %s", bundle.get("model_name", "?"), args.model)

    df = pd.read_csv(args.input)
    log.info("loaded %s: %d rows, %d columns", args.input, *df.shape)

    preds = predict_frame(bundle, df)

    if "target" in df.columns:
        log.info("RMSE against provided target: %.6g", rmse(df["target"].to_numpy(), preds))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": preds}, index=df.index).to_csv(args.output, index=False)
    log.info("wrote %d predictions to %s", len(preds), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
