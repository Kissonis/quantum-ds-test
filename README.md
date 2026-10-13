# DS Engineer — test assignment

Three independent tasks, one per directory:

| # | Task | Directory | Highlights |
|---|------|-----------|------------|
| 1 | Counting islands (classical algorithms) | [`task1_islands/`](task1_islands/) | iterative flood fill, O(M·N), 4-connectivity, 23 tests |
| 2 | Regression on tabular data (general ML) | [`task2_regression/`](task2_regression/) | EDA notebook, `train.py` / `predict.py`, predictions for `hidden_test.csv`, 7 tests |
| 3 | MNIST classifier (OOP) | [`task3_mnist/`](task3_mnist/) | `DigitClassificationInterface`, 3 models, `DigitClassifier` facade, 33 tests |

Each directory has its own README with details, usage and design notes.

## Setup

Python 3.10+ (developed and tested on 3.13). One virtual environment covers all
three tasks:

```bash
git clone <this repo>
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` at the root installs everything. Per-task files
(`task2_regression/requirements.txt`, `task3_mnist/requirements.txt`) list only
what that task needs. Task 1 has no third-party dependencies.

PyTorch is installed from the CPU wheel index to keep the download small; see
`task3_mnist/requirements.txt` if you want a GPU build.

**Data for task 2.** The CSV files are provided with the assignment and are not
stored in git. Copy `train.csv` and `hidden_test.csv` into
`task2_regression/data/` before running `train.py`, `predict.py` or the notebook
(see [`task2_regression/data/README.md`](task2_regression/data/README.md)).
The committed `predictions.csv` and fitted model do not need them.

## Quick start

```bash
# Task 1
printf '3 4\n0 0 0 1\n0 0 1 1\n0 1 0 1\n' | python task1_islands/islands.py     # -> 2

# Task 2: first copy the provided train.csv and hidden_test.csv into task2_regression/data/
cd task2_regression
python train.py                      # fits the model, prints CV RMSE, saves models/model.joblib
python predict.py                    # writes predictions.csv for data/hidden_test.csv
cd ..

# Task 3
cd task3_mnist
python -c "import numpy as np; from mnist_classifier import DigitClassifier; \
print(DigitClassifier('cnn').predict(np.zeros((28, 28, 1), dtype=np.float32)))"
cd ..
```

## Tests

```bash
pytest -q
```

Runs the test suites of all three tasks (63 tests, a few seconds). The task 2
tests use small synthetic data, so they do not need the CSV files.

## Layout

```
.
├── README.md
├── requirements.txt
├── task1_islands/
│   ├── islands.py
│   ├── tests/test_islands.py
│   └── README.md
├── task2_regression/
│   ├── data/                 train.csv, hidden_test.csv (not in git, see data/README.md)
│   ├── notebooks/eda.ipynb   exploratory data analysis (executed, with outputs)
│   ├── features.py           feature engineering shared by train/predict
│   ├── train.py              training + cross-validation, saves the model
│   ├── predict.py            inference, writes predictions.csv
│   ├── models/model.joblib   fitted final model (+ metrics json)
│   ├── predictions.csv       predictions for hidden_test.csv
│   ├── tests/test_regression.py
│   ├── requirements.txt
│   └── README.md
└── task3_mnist/
    ├── mnist_classifier/     package: interface, 3 models, DigitClassifier
    ├── tests/test_classifier.py
    ├── requirements.txt
    └── README.md
```
