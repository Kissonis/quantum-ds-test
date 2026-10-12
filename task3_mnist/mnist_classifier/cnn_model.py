"""Convolutional neural network classifier (PyTorch).

PyTorch is imported lazily, the first time a :class:`CNNClassifier` is
created, so that ``import mnist_classifier`` and the lightweight models
(``"rf"``, ``"rand"``) do not pay for loading torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .interface import NUM_CLASSES, DigitClassificationInterface

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from torch import nn


def build_network() -> "nn.Module":
    """Return a fresh small CNN. Input ``(B, 1, 28, 28)``, output ``(B, 10)`` logits.

    Two conv blocks followed by a small MLP head.  Defined inside a function
    so that torch is only imported when a network is actually needed.
    """
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 28 -> 14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 14 -> 7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


class CNNClassifier(DigitClassificationInterface):
    """CNN digit classifier.

    Expects the common ``(28, 28, 1)`` image and converts it internally to the
    ``(1, 1, 28, 28)`` tensor PyTorch wants (batch, channels, height, width).

    Parameters
    ----------
    weights_path:
        Optional path to a ``state_dict`` saved with ``torch.save``.  When
        omitted the network keeps its random initialisation, which is enough
        for this exercise since training is out of scope.
    seed:
        Seed for the random initialisation, for reproducible results.  It is
        applied inside :func:`torch.random.fork_rng`, so creating a classifier
        does not change the caller's global torch random state.
    """

    name = "cnn"

    def __init__(self, weights_path: Optional[str] = None, seed: int = 0) -> None:
        import torch

        self._torch = torch
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._model: Any = build_network()
        if weights_path is not None:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, image: np.ndarray) -> int:
        image = self.validate_image(image)
        torch = self._torch
        # (28, 28, 1) -> (1, 1, 28, 28): add batch dim, move channel first.
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
        return int(logits.argmax(dim=1).item())
