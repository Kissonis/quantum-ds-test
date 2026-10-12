"""Common interface for all MNIST digit classifiers.

Every model used by :class:`mnist_classifier.DigitClassifier` must implement
:class:`DigitClassificationInterface`.  The contract is deliberately small so
that new models (SVM, kNN, a transformer, ...) can be plugged in by other
developers without touching the rest of the package.

Contract
--------
* ``predict(image)`` receives a single image as a NumPy array of shape
  ``(28, 28, 1)`` and returns one integer in ``0..9``.
* Pixel values are intensities, either in ``[0, 1]`` or in ``[0, 255]`` (the
  raw MNIST format).  :meth:`DigitClassificationInterface.validate_image`
  rescales ``[0, 255]`` input to ``[0, 1]``, so every model always works with
  the ``[0, 1]`` range.
* ``train(...)`` is reserved for future work and raises
  :class:`NotImplementedError` by default.
* Model-specific input preparation (flattening, cropping, tensor conversion)
  is an implementation detail hidden inside each model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

#: Shape of the image every classifier accepts: height x width x channels.
IMAGE_SHAPE = (28, 28, 1)

#: Number of classes (digits 0..9).
NUM_CLASSES = 10

#: Largest accepted pixel value (raw 8-bit MNIST intensity).
MAX_PIXEL = 255.0


class DigitClassificationInterface(ABC):
    """Abstract base class for a single-digit classifier.

    Subclasses must implement :meth:`predict`.  Use :meth:`validate_image`
    at the start of ``predict`` to enforce the common input format.
    """

    #: Short name used to select the model, e.g. ``"cnn"``.
    name: str = ""

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """Classify one image.

        Parameters
        ----------
        image:
            Array of shape ``(28, 28, 1)``.

        Returns
        -------
        int
            Predicted digit, ``0..9``.
        """

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Training is intentionally out of scope for this exercise."""
        raise NotImplementedError(
            f"{type(self).__name__}.train() is not implemented"
        )

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Validate ``image`` and return it as float32 in the ``[0, 1]`` range.

        Accepts intensities in ``[0, 1]`` or in ``[0, 255]``.  If any value is
        greater than 1 the image is treated as 8-bit and divided by 255.  The
        caller's array is never modified.

        Raises
        ------
        TypeError
            If ``image`` is not a NumPy array.
        ValueError
            If the shape is not ``(28, 28, 1)``, or values are NaN/inf,
            negative or greater than 255.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}")
        if image.shape != IMAGE_SHAPE:
            raise ValueError(f"image must have shape {IMAGE_SHAPE}, got {image.shape}")
        image = image.astype(np.float32, copy=False)
        if not np.isfinite(image).all():
            raise ValueError("image contains NaN or infinite values")
        low, high = float(image.min()), float(image.max())
        if low < 0 or high > MAX_PIXEL:
            raise ValueError(
                f"pixel values must be in [0, 1] or [0, {MAX_PIXEL:g}], got [{low:g}, {high:g}]"
            )
        if high > 1.0:
            image = image / MAX_PIXEL  # new array: caller's data stays intact
        return image
