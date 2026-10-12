"""Tests for the DigitClassifier facade and the three model implementations."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mnist_classifier import (  # noqa: E402
    ALGORITHMS,
    CNNClassifier,
    DigitClassificationInterface,
    DigitClassifier,
    RandomDigitClassifier,
    RandomForestDigitClassifier,
)
from mnist_classifier.random_model import CROP_SIZE  # noqa: E402


@pytest.fixture
def image():
    rng = np.random.default_rng(42)
    return rng.random((28, 28, 1), dtype=np.float32)


# --- interface --------------------------------------------------------------

def test_interface_is_abstract():
    with pytest.raises(TypeError):
        DigitClassificationInterface()  # type: ignore[abstract]


@pytest.mark.parametrize("cls", [CNNClassifier, RandomForestDigitClassifier, RandomDigitClassifier])
def test_models_implement_interface(cls):
    assert issubclass(cls, DigitClassificationInterface)
    assert cls.name in ALGORITHMS


def test_validate_image_rejects_bad_inputs():
    with pytest.raises(TypeError):
        DigitClassificationInterface.validate_image([[0] * 28] * 28)
    with pytest.raises(ValueError):
        DigitClassificationInterface.validate_image(np.zeros((28, 28)))
    with pytest.raises(ValueError):
        DigitClassificationInterface.validate_image(np.zeros((1, 28, 28, 1)))


# --- facade -----------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
def test_same_io_contract_for_every_algorithm(algorithm, image):
    clf = DigitClassifier(algorithm, seed=0)
    digit = clf.predict(image)
    assert type(digit) is int
    assert 0 <= digit <= 9


@pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
def test_predict_rejects_wrong_shape(algorithm):
    clf = DigitClassifier(algorithm, seed=0)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((28, 28)))


@pytest.mark.parametrize("algorithm", ["cnn", "rf", "rand"])
def test_train_raises_not_implemented(algorithm):
    clf = DigitClassifier(algorithm, seed=0)
    with pytest.raises(NotImplementedError):
        clf.train()
    with pytest.raises(NotImplementedError):
        clf.model.train()


def test_unknown_algorithm():
    with pytest.raises(ValueError):
        DigitClassifier("svm")


def test_algorithm_name_is_case_insensitive(image):
    assert DigitClassifier(" CNN ", seed=0).algorithm == "cnn"


def test_cnn_is_deterministic_for_fixed_seed(image):
    a = DigitClassifier("cnn", seed=1).predict(image)
    b = DigitClassifier("cnn", seed=1).predict(image)
    assert a == b


def test_register_new_model(image):
    class ConstantClassifier(DigitClassificationInterface):
        name = "const"

        def predict(self, img):
            self.validate_image(img)
            return 7

    DigitClassifier.register(ConstantClassifier)
    try:
        assert DigitClassifier("const").predict(image) == 7
    finally:
        ALGORITHMS.pop("const", None)


def test_register_rejects_non_interface():
    with pytest.raises(TypeError):
        DigitClassifier.register(object)  # type: ignore[arg-type]


def test_register_works_as_decorator(image):
    try:
        @DigitClassifier.register
        class FiveClassifier(DigitClassificationInterface):
            name = "five"

            def predict(self, img):
                return 5

        assert DigitClassifier("five").predict(image) == 5
    finally:
        ALGORITHMS.pop("five", None)


def test_register_refuses_to_shadow_existing_model():
    class FakeCNN(DigitClassificationInterface):
        name = "cnn"

        def predict(self, img):
            return 0

    with pytest.raises(ValueError, match="already registered"):
        DigitClassifier.register(FakeCNN)
    assert ALGORITHMS["cnn"] is CNNClassifier


def test_register_override_replaces_model(image):
    original = ALGORITHMS["rand"]
    try:
        @DigitClassifier.register(override=True)
        class ZeroRandom(DigitClassificationInterface):
            name = "rand"

            def predict(self, img):
                return 0

        assert DigitClassifier("rand").predict(image) == 0
    finally:
        ALGORITHMS["rand"] = original


# --- pixel range contract ---------------------------------------------------

def test_validate_image_rescales_8bit_input():
    raw = np.full((28, 28, 1), 255, dtype=np.uint8)
    out = DigitClassificationInterface.validate_image(raw)
    assert out.dtype == np.float32
    assert out.max() == pytest.approx(1.0)
    assert raw.max() == 255  # caller's array untouched


def test_validate_image_keeps_unit_range_input(image):
    out = DigitClassificationInterface.validate_image(image)
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize("value", [-0.1, 256.0, np.nan, np.inf])
def test_validate_image_rejects_out_of_range(value):
    img = np.zeros((28, 28, 1), dtype=np.float32)
    img[0, 0, 0] = value
    with pytest.raises(ValueError):
        DigitClassificationInterface.validate_image(img)


@pytest.mark.parametrize("algorithm", ["cnn", "rf"])
def test_8bit_and_unit_inputs_give_same_prediction(algorithm, image):
    clf = DigitClassifier(algorithm)
    assert clf.predict(image) == clf.predict(image * 255.0)


# --- side effects -----------------------------------------------------------

def test_cnn_init_does_not_touch_global_torch_rng():
    import torch

    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    CNNClassifier(seed=7)
    assert torch.equal(torch.rand(3), expected)


def test_importing_package_does_not_import_torch():
    import subprocess

    code = (
        "import sys; sys.path.insert(0, %r); import mnist_classifier; "
        "mnist_classifier.DigitClassifier('rand'); print('torch' in sys.modules)"
    ) % os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- random model specifics ---------------------------------------------------

def test_center_crop_shape_and_position():
    img = np.arange(28 * 28, dtype=np.float32).reshape(28, 28, 1)
    crop = RandomDigitClassifier.center_crop(img)
    assert crop.shape == (CROP_SIZE, CROP_SIZE)
    assert crop[0, 0] == img[9, 9, 0]
    assert crop[-1, -1] == img[18, 18, 0]
