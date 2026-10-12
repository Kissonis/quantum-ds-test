# Task 3 — MNIST classifier (OOP)

A `DigitClassifier` with a pluggable backend. Three models are provided; new
ones can be added by implementing a single interface.

## Design

```
DigitClassificationInterface  (abstract)
├── CNNClassifier                 "cnn"   PyTorch CNN, input tensor 28x28x1
├── RandomForestDigitClassifier   "rf"    scikit-learn RF, input flat vector of 784
└── RandomDigitClassifier         "rand"  random digit, input 10x10 centre crop

DigitClassifier(algorithm)  → picks a model from the registry, same predict() for all
```

* **`DigitClassificationInterface`** (`mnist_classifier/interface.py`) — abstract
  base class. Subclasses implement `predict(image) -> int`; `train()` raises
  `NotImplementedError` by default. `validate_image()` enforces the common
  input contract (see below).
* **Three model classes**, one per module. Each takes the common `(28, 28, 1)`
  image and performs its own preprocessing internally (channel-first tensor,
  flattening to 784, 10x10 centre crop), so callers never need to know which
  model is behind the facade.
* **`DigitClassifier`** (`mnist_classifier/classifier.py`) — takes the algorithm
  name (`cnn`, `rf`, `rand`), instantiates the model and delegates `predict()`.
  The input/output contract is identical regardless of the algorithm.
  `DigitClassifier.register(cls)` lets other developers add models.

## Input contract

* Shape `(28, 28, 1)`, a NumPy array.
* Pixel intensities in `[0, 1]` **or** `[0, 255]` (raw MNIST). If any value is
  above 1 the image is treated as 8-bit and divided by 255, so every model
  always sees `[0, 1]`. The caller's array is never modified.
* NaN/inf, negative values and values above 255 raise `ValueError`.

## Implementation notes

* PyTorch is imported lazily, when a `CNNClassifier` is first created, so
  `import mnist_classifier` and the `rf` / `rand` models start fast.
* The CNN `seed` is applied inside `torch.random.fork_rng`, so creating a
  classifier does not change the caller's global torch random state.

Training is out of scope by the task statement: `train()` always raises
`NotImplementedError`. The CNN uses its random initialisation (a `weights_path`
can be passed to load a trained `state_dict`). scikit-learn does not allow
`predict()` on an unfitted estimator, so the Random Forest is fitted on a tiny
synthetic dataset in `__init__` purely to make the object usable — this is
documented in the class and its predictions are not meaningful. A fitted model
can be loaded via `model_path`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r task3_mnist/requirements.txt
```

## Usage

```python
import numpy as np
from mnist_classifier import DigitClassifier

image = np.random.rand(28, 28, 1).astype(np.float32)

for algo in ("cnn", "rf", "rand"):
    clf = DigitClassifier(algo)
    print(algo, clf.predict(image))   # -> int in 0..9
```

Run from the `task3_mnist` directory (or add it to `PYTHONPATH`).

## Adding a new model

```python
from mnist_classifier import DigitClassifier, DigitClassificationInterface

@DigitClassifier.register
class SVMClassifier(DigitClassificationInterface):
    name = "svm"

    def predict(self, image):
        image = self.validate_image(image)   # shape check + [0, 1] range
        ...
        return digit
```

The decorator runs when the module is imported, so import the module (or add
it to `mnist_classifier/__init__.py`) before calling `DigitClassifier("svm")`.

Registering a name that is already taken raises `ValueError`, so a built-in
model cannot be replaced by accident. To replace one on purpose:

```python
@DigitClassifier.register(override=True)
class BetterCNN(DigitClassificationInterface):
    name = "cnn"
    ...
```

## Tests

```bash
pytest task3_mnist/tests -q
```
