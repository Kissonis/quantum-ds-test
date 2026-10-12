"""MNIST digit classification with pluggable models.

Public API::

    from mnist_classifier import DigitClassifier
    clf = DigitClassifier("cnn")      # or "rf", "rand"
    digit = clf.predict(image)        # image: np.ndarray of shape (28, 28, 1)
"""

from .classifier import ALGORITHMS, DigitClassifier
from .cnn_model import CNNClassifier
from .interface import IMAGE_SHAPE, NUM_CLASSES, DigitClassificationInterface
from .random_model import RandomDigitClassifier
from .rf_model import RandomForestDigitClassifier

__all__ = [
    "ALGORITHMS",
    "IMAGE_SHAPE",
    "NUM_CLASSES",
    "DigitClassifier",
    "DigitClassificationInterface",
    "CNNClassifier",
    "RandomForestDigitClassifier",
    "RandomDigitClassifier",
]
