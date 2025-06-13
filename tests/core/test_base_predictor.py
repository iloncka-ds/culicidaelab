import numpy as np
import pytest
from unittest.mock import Mock

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.config_manager import ConfigManager


class DummyPredictor(BasePredictor):
    def _load_model(self):
        self.dummy_loaded = True

    def predict(self, input_data: np.ndarray):
        return float(np.sum(input_data))

    def visualize(self, input_data: np.ndarray, predictions, save_path=None):
        return input_data

    def evaluate(self, input_data: np.ndarray, ground_truth):
        pred = self.predict(input_data)
        return {"accuracy": float(pred == ground_truth)}


@pytest.fixture
def dummy_config_manager(tmp_path):
    return Mock(spec=ConfigManager)


@pytest.fixture
def dummy_predictor(dummy_config_manager, tmp_path):
    return DummyPredictor(
        model_path=tmp_path / "dummy.pth",
        config_manager=dummy_config_manager,
    )


def test_init_and_load_model(dummy_predictor):
    assert not dummy_predictor.model_loaded
    dummy_predictor.load_model()
    assert dummy_predictor.model_loaded
    assert hasattr(dummy_predictor, "dummy_loaded") and dummy_predictor.dummy_loaded


def test_predict_and_call(dummy_predictor):
    arr = np.ones((2, 2))
    pred = dummy_predictor.predict(arr)
    assert pred == 4.0
    dummy_predictor.model_loaded = False
    pred2 = dummy_predictor(arr)
    assert pred2 == 4.0
    assert dummy_predictor.model_loaded


def test_visualize(dummy_predictor):
    arr = np.zeros((2, 2))
    vis = dummy_predictor.visualize(arr, predictions=None)
    assert np.array_equal(vis, arr)


def test_evaluate(dummy_predictor):
    arr = np.ones((2, 2))
    metrics = dummy_predictor.evaluate(arr, 4.0)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_predict_batch(dummy_predictor):
    arrs = [np.ones((2, 2)), np.zeros((2, 2))]
    preds = dummy_predictor.predict_batch(arrs, num_workers=1, batch_size=1)
    assert sorted(preds) == [0.0, 4.0]


def test_evaluate_batch(dummy_predictor):
    arrs = [np.ones((2, 2)), np.zeros((2, 2))]
    gts = [4.0, 0.0]
    metrics = dummy_predictor.evaluate_batch(arrs, gts, num_workers=1, batch_size=1)
    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0
    assert "accuracy_std" in metrics


def test_aggregate_metrics(dummy_predictor):
    metrics_list = [{"accuracy": 1.0}, {"accuracy": 0.0}]
    agg = dummy_predictor._aggregate_metrics(metrics_list)
    assert agg["accuracy"] == 0.5
    assert "accuracy_std" in agg
