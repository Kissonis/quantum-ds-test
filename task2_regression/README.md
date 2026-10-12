# Task 2 — Regression on tabular data

Predict `target` from 53 anonymised features. Metric: RMSE.

## TL;DR

Exploratory analysis ([`notebooks/eda.ipynb`](notebooks/eda.ipynb)) shows the
data is synthetic and the target is a deterministic function of two features:

```
target = feature_6 ** 2 + feature_7
```

The other 51 columns are noise. The final model is a linear regression on the
two engineered features `[6², 7]`; it reproduces the target to machine precision.
A LightGBM model on all raw features is kept as a generic baseline.

| model | inputs | 5-fold CV RMSE |
|---|---|---|
| **LinearRegression** (final) | `6²`, `7` | **1.4e-13** |
| LightGBM (baseline) | all 53 raw features | 0.009 (3-fold) |

Predictions for `data/hidden_test.csv` are in [`predictions.csv`](predictions.csv)
(one column `target`, same row order as the input).

## How the formula was found

1. All Pearson correlations with the target are below 0.013 in absolute value,
   so a linear model on raw features would be useless.
2. A quick gradient-boosting model nevertheless reaches a low error, and its
   feature importance says feature `6` explains almost everything, `7` the rest.
3. A scatter of `target` vs `6` is a parabola (hence zero *linear* correlation —
   a symmetric function of a symmetric variable). `target` vs `6²` is a line of slope 1.
4. The residual `target − 6²` lies in `[0, 1]` and correlates 1.0000 with feature `7`.
5. `target − 6² − 7` has RMSE ≈ 1e-14 on the whole training set.

The notebook walks through each step with plots.

## Files

```
data/train.csv, data/hidden_test.csv   provided data (not in git, see data/README.md)
notebooks/eda.ipynb                    EDA, executed with outputs
features.py                            engineered features shared by train/predict
train.py                               training, CV, saves models/model.joblib (+ .metrics.json)
predict.py                             inference, writes predictions.csv
models/model.joblib                    fitted final (linear) model
predictions.csv                        predictions for hidden_test.csv
tests/test_regression.py               unit + end-to-end tests on synthetic data
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r task2_regression/requirements.txt
cd task2_regression
```

The datasets are not stored in git. Copy `train.csv` and `hidden_test.csv` into
`data/` first (see [`data/README.md`](data/README.md)).

## Training

```bash
python train.py                                   # linear model (default)
python train.py --model lgbm --output models/lgbm.joblib
python train.py --folds 10 --holdout 0.1 --seed 7
python train.py --help
```

`train.py` reports a hold-out RMSE and K-fold cross-validation RMSE, refits on
the full training set, and saves a bundle (`model`, `feature_mode`, metrics)
to `--output`. Metrics are also written next to the model as JSON.

## Inference

```bash
python predict.py                                                   # data/hidden_test.csv -> predictions.csv
python predict.py --model models/lgbm.joblib --output predictions_lgbm.csv
python predict.py --input some_file.csv --output out.csv
```

The bundle records which features the model expects, so `predict.py` always
rebuilds exactly the same inputs as training. If the input file contains a
`target` column the script also prints the RMSE.

## Tests

```bash
pytest task2_regression/tests -q
```

The tests build a small synthetic frame with the same layout as `train.csv`
(columns `"0"`..`"52"`, `target = f6² + f7`), so they do not need the real data.
They cover the feature engineering, the full `train.py` → `predict.py` round trip
for the linear model (coefficients recovered as `[1, 1]`), the LightGBM path and
the error handling for missing files.

## Reproducing the notebook

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
```

## Notes and caveats

* The exact relationship holds on the provided data. The linear model is the
  right choice *because* EDA established the functional form; on data generated
  differently it would not adapt, whereas the LightGBM baseline would.
* `colsample_bytree` was deliberately left at 1.0 for LightGBM: with 51 noise
  features, randomly dropping one of the two informative columns from a tree
  costs far more than the regularisation gains (RMSE 0.13 vs 0.009).
