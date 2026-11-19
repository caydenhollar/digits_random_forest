import json

import joblib
import numpy as np
import pytest

from digits import predict, read_pixels, split_data, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    path = tmp_path_factory.mktemp("forest")
    model, metrics = train(path)
    return path, model, metrics


def test_beats_baseline_on_held_out_digits(trained):
    _, _, metrics = trained
    assert metrics["accuracy"] > 0.93
    assert metrics["accuracy"] > metrics["baseline_accuracy"] + 0.75
    assert sum(map(sum, metrics["confusion_matrix"])) == metrics["test_samples"]


def test_saved_model_preserves_predictions(trained):
    path, model, metrics = trained
    _, x_test, _, _ = split_data()
    loaded = joblib.load(path / "forest.joblib")
    np.testing.assert_array_equal(model.predict(x_test), loaded.predict(x_test))
    assert json.loads((path / "metrics.json").read_text()) == metrics


def test_pixel_file_round_trip(tmp_path, trained):
    path = tmp_path / "digit.csv"
    _, x_test, _, _ = split_data()
    np.savetxt(path, x_test[0].reshape(8, 8), delimiter=",")
    pixels = read_pixels(path)
    np.testing.assert_array_equal(pixels[0], x_test[0])
    result = predict(trained[1], pixels)
    assert 0 <= result["digit"] <= 9
    assert 0 <= result["vote_share"] <= 1


@pytest.mark.parametrize("pixels", [np.zeros((7, 8)), np.full((8, 8), 17),
                                    np.full((8, 8), -1), np.full((8, 8), np.nan)])
def test_rejects_invalid_pixels(tmp_path, pixels):
    path = tmp_path / "bad.csv"
    np.savetxt(path, pixels, delimiter=",")
    with pytest.raises(ValueError):
        read_pixels(path)


def test_split_is_repeatable_and_has_every_class():
    first, second = split_data(), split_data()
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert set(first[3]) == set(range(10))
