"""Random Forest classifier (scikit-learn)."""

from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .interface import IMAGE_SHAPE, NUM_CLASSES, DigitClassificationInterface

#: Length of the flattened image: 28 * 28.
FLAT_SIZE = IMAGE_SHAPE[0] * IMAGE_SHAPE[1]


class RandomForestDigitClassifier(DigitClassificationInterface):
    """Random Forest digit classifier.

    Expects the common ``(28, 28, 1)`` image and flattens it internally into
    the 1-D vector of length 784 that the forest consumes.

    Parameters
    ----------
    model_path:
        Optional path to a fitted ``RandomForestClassifier`` saved with
        ``joblib.dump``.
    seed:
        Random seed used both for the forest and for the placeholder data.

    Notes
    -----
    scikit-learn refuses to call ``predict`` on an unfitted estimator.  Since
    training is explicitly out of scope for this exercise, when no
    ``model_path`` is given the forest is fitted on a tiny **synthetic**
    dataset (random noise with random labels) inside ``__init__``.  This only
    makes the object usable end-to-end; its predictions are meaningless and
    it must be replaced by a properly trained model for real use.
    """

    name = "rf"

    def __init__(self, model_path: Optional[str] = None, seed: int = 0) -> None:
        if model_path is not None:
            self._model = joblib.load(model_path)
        else:
            self._model = RandomForestClassifier(n_estimators=10, random_state=seed)
            self._fit_placeholder(seed)

    def _fit_placeholder(self, seed: int) -> None:
        """Fit on random data so that ``predict`` works without real training."""
        rng = np.random.default_rng(seed)
        n_samples = 10 * NUM_CLASSES
        x = rng.random((n_samples, FLAT_SIZE), dtype=np.float32)
        # Every class appears at least once so predict() can output any digit.
        y = np.tile(np.arange(NUM_CLASSES), n_samples // NUM_CLASSES)
        self._model.fit(x, y)

    def predict(self, image: np.ndarray) -> int:
        image = self.validate_image(image)
        flat = image.reshape(1, FLAT_SIZE)  # (28, 28, 1) -> (1, 784)
        return int(self._model.predict(flat)[0])
