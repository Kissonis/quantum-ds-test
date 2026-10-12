"""Random "classifier" used as a trivial baseline / stub."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .interface import NUM_CLASSES, DigitClassificationInterface

#: Side of the square crop taken from the centre of the image.
CROP_SIZE = 10


class RandomDigitClassifier(DigitClassificationInterface):
    """Returns a random digit regardless of the input.

    Expects the common ``(28, 28, 1)`` image and crops the central
    ``10 x 10`` patch internally, as the task specifies for this model.
    The crop is computed (and shape-checked) even though its content is not
    used, so the preprocessing path is exercised the same way as in a real
    model.

    Parameters
    ----------
    seed:
        Optional seed for reproducible "predictions".
    """

    name = "rand"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def center_crop(image: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
        """Return the central ``size x size`` patch of a ``(H, W, 1)`` image as ``(size, size)``."""
        h, w = image.shape[:2]
        top = (h - size) // 2
        left = (w - size) // 2
        return image[top:top + size, left:left + size, 0]

    def predict(self, image: np.ndarray) -> int:
        image = self.validate_image(image)
        crop = self.center_crop(image)
        assert crop.shape == (CROP_SIZE, CROP_SIZE)
        return int(self._rng.integers(0, NUM_CLASSES))
