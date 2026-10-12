# Data

The datasets are provided with the assignment and are not stored in git
(`train.csv` is about 75 MB).

Put the two files here before running `train.py`, `predict.py` or the notebook:

```
task2_regression/data/
├── train.csv         90 000 rows: feature columns "0".."52" + "target"
└── hidden_test.csv   10 000 rows: feature columns "0".."52"
```

Predictions for `hidden_test.csv` are already committed in
[`../predictions.csv`](../predictions.csv), and the fitted model in
[`../models/model.joblib`](../models/model.joblib), so neither file is needed
just to inspect the results. The unit tests use synthetic data and do not need
them either.
