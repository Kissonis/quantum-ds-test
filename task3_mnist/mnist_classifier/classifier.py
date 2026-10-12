"""Facade that selects a concrete model by name."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import numpy as np

from .cnn_model import CNNClassifier
from .interface import DigitClassificationInterface
from .random_model import RandomDigitClassifier
from .rf_model import RandomForestDigitClassifier

#: Registry of available algorithms.  Register a new model here (or via
#: :meth:`DigitClassifier.register`) to make it selectable by name.
ALGORITHMS: Dict[str, Type[DigitClassificationInterface]] = {
    CNNClassifier.name: CNNClassifier,
    RandomForestDigitClassifier.name: RandomForestDigitClassifier,
    RandomDigitClassifier.name: RandomDigitClassifier,
}


class DigitClassifier:
    """Digit classifier with a pluggable backend.

    Parameters
    ----------
    algorithm:
        One of ``"cnn"``, ``"rf"``, ``"rand"`` (see :data:`ALGORITHMS`).
    **model_kwargs:
        Forwarded to the selected model's constructor (e.g. ``weights_path``).

    Examples
    --------
    >>> import numpy as np
    >>> clf = DigitClassifier("rand", seed=0)
    >>> digit = clf.predict(np.zeros((28, 28, 1), dtype=np.float32))
    >>> 0 <= digit <= 9
    True
    """

    def __init__(self, algorithm: str, **model_kwargs: Any) -> None:
        key = algorithm.lower().strip()
        if key not in ALGORITHMS:
            raise ValueError(
                f"unknown algorithm {algorithm!r}; available: {sorted(ALGORITHMS)}"
            )
        self.algorithm = key
        self.model: DigitClassificationInterface = ALGORITHMS[key](**model_kwargs)

    def predict(self, image: np.ndarray) -> int:
        """Classify one ``(28, 28, 1)`` image and return a digit ``0..9``.

        The input and output format is identical for every algorithm; any
        model-specific preprocessing happens inside the model.
        """
        return self.model.predict(image)

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Training is out of scope; always raises ``NotImplementedError``."""
        raise NotImplementedError("DigitClassifier.train() is not implemented")

    @classmethod
    def register(
        cls,
        model_cls: Optional[Type[DigitClassificationInterface]] = None,
        *,
        override: bool = False,
    ) -> Any:
        """Register an additional model class under its ``name`` attribute.

        Works as a plain call or as a decorator, with or without arguments::

            @DigitClassifier.register
            class SVMClassifier(DigitClassificationInterface):
                name = "svm"
                ...

            @DigitClassifier.register(override=True)   # replace an existing "svm"
            class BetterSVM(DigitClassificationInterface):
                name = "svm"
                ...

        Parameters
        ----------
        model_cls:
            Class implementing :class:`DigitClassificationInterface`.
        override:
            Allow replacing a model that is already registered under the same
            name.  Off by default so a built-in model cannot be shadowed by
            accident.

        Raises
        ------
        TypeError
            If the class does not implement the interface.
        ValueError
            If ``name`` is empty, or already taken and ``override`` is False.
        """

        def _register(model: Type[DigitClassificationInterface]) -> Type[DigitClassificationInterface]:
            if not (isinstance(model, type) and issubclass(model, DigitClassificationInterface)):
                raise TypeError("model must implement DigitClassificationInterface")
            if not model.name:
                raise ValueError("model class must define a non-empty `name`")
            existing = ALGORITHMS.get(model.name)
            if existing is not None and existing is not model and not override:
                raise ValueError(
                    f"algorithm {model.name!r} is already registered by {existing.__name__}; "
                    "pass override=True to replace it"
                )
            ALGORITHMS[model.name] = model
            return model

        if model_cls is None:  # used as @register(...) with arguments
            return _register
        return _register(model_cls)

    def __repr__(self) -> str:
        return f"DigitClassifier(algorithm={self.algorithm!r})"
