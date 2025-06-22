import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future
import logging

# Mock Pydantic config model if not available
try:
    from culicidaelab.core.config_models import PredictorConfig
except ImportError:
    PredictorConfig = type("PredictorConfig", (object,), {})  # type: ignore[assignment, misc]
    import sys

    sys.modules["culicidaelab.core.config_models"] = MagicMock(PredictorConfig=PredictorConfig)

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.core.model_weights_manager import ModelWeightsManager


class DummyPredictor(BasePredictor):
    """Test implementation of BasePredictor"""

    def __init__(self, *args, **kwargs):
        self.dummy_loaded = False
        self._model = None
        super().__init__(*args, **kwargs)

    def _load_model(self):
        """Mock model loading"""
        self.dummy_loaded = True
        self._model = "dummy_model"

    def predict(self, input_data: np.ndarray, **kwargs):
        """Return sum of input data as prediction"""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")
        return float(np.sum(input_data))

    def visualize(self, input_data: np.ndarray, predictions, save_path=None):
        """Return input data as visualization"""
        return input_data

    def _evaluate_from_prediction(self, prediction, ground_truth):
        """Evaluate prediction against ground truth"""
        accuracy = 1.0 if prediction == ground_truth else 0.0
        return {"accuracy": accuracy, "mse": float((prediction - ground_truth) ** 2)}


class FailingPredictor(BasePredictor):
    """Test predictor that fails to load"""

    def _load_model(self):
        raise Exception("Failed to load model")

    def predict(self, input_data: np.ndarray, **kwargs):
        return 0.0

    def visualize(self, input_data: np.ndarray, predictions, save_path=None):
        return input_data

    def _evaluate_from_prediction(self, prediction, ground_truth):
        return {"accuracy": 0.0}


@pytest.fixture
def dummy_settings(tmp_path):
    """Create mock settings with a Pydantic-like config object."""
    settings = Mock(spec=Settings)

    # Create a dummy model file
    dummy_weights = tmp_path / "dummy.pth"
    dummy_weights.touch()

    # Create a mock ModelWeightsManager
    mock_weights_manager = Mock(spec=ModelWeightsManager)
    mock_weights_manager.ensure_weights.return_value = dummy_weights

    # Create a mock Pydantic PredictorConfig object
    predictor_config_dict = {
        "repository_id": "dummy/repo",
        "filename": "model.pth",
        "target_": "dummy.module.DummyModel",
        "params": {"param1": "value1"},
        "model_path": str(dummy_weights),
        "device": "cpu",
        "confidence": 0.5,
    }
    # Use MagicMock to allow attribute setting
    mock_predictor_config = MagicMock(spec=PredictorConfig)
    for key, value in predictor_config_dict.items():
        setattr(mock_predictor_config, key, value)
    mock_predictor_config.model_dump.return_value = predictor_config_dict

    # Mock get_config method to return the Pydantic-like object
    def get_config(path=None, default=None):
        if path in ["predictors.dummy", "predictors.failing"]:
            return mock_predictor_config
        return default

    settings.get_config.side_effect = get_config

    # Mock the ModelWeightsManager
    with patch("culicidaelab.core.base_predictor.ModelWeightsManager", return_value=mock_weights_manager):
        yield settings


@pytest.fixture
def dummy_predictor(dummy_settings):
    """Create a dummy predictor instance"""
    return DummyPredictor(
        settings=dummy_settings,
        predictor_type="dummy",
    )


@pytest.fixture
def loaded_predictor(dummy_settings):
    """Create a dummy predictor with model pre-loaded"""
    return DummyPredictor(settings=dummy_settings, predictor_type="dummy", load_model=True)


def test_init_basic(dummy_predictor):
    """Test basic initialization"""
    assert dummy_predictor.predictor_type == "dummy"
    assert not dummy_predictor.model_loaded
    assert dummy_predictor._model is None
    assert dummy_predictor.model_path.name == "dummy.pth"


def test_init_with_load_model(dummy_settings):
    """Test initialization with immediate model loading"""
    predictor = DummyPredictor(settings=dummy_settings, predictor_type="dummy", load_model=True)
    assert predictor.model_loaded
    assert predictor.dummy_loaded


def test_init_missing_config(dummy_settings):
    """Test initialization with missing predictor config"""
    # Configure mock to return None, which BasePredictor should handle
    dummy_settings.get_config.side_effect = lambda path=None, default=None: None

    with pytest.raises(ValueError, match="Configuration for predictor 'missing' not found or is invalid"):
        DummyPredictor(settings=dummy_settings, predictor_type="missing")


def test_properties(dummy_predictor):
    """Test property accessors"""
    assert isinstance(dummy_predictor.model_path, Path)
    assert dummy_predictor.model_path.name == "dummy.pth"

    config = dummy_predictor.config
    assert config.repository_id == "dummy/repo"
    assert config.params["param1"] == "value1"

    assert not dummy_predictor.model_loaded


def test_load_model_success(dummy_predictor):
    """Test successful model loading"""
    assert not dummy_predictor.model_loaded

    dummy_predictor.load_model()

    assert dummy_predictor.model_loaded
    assert dummy_predictor.dummy_loaded
    assert dummy_predictor._model == "dummy_model"


def test_load_model_already_loaded(loaded_predictor, caplog):
    """Test loading model when already loaded"""
    with caplog.at_level(logging.INFO):
        loaded_predictor.load_model()

    assert "already loaded" in caplog.text


def test_load_model_failure(dummy_settings):
    """Test model loading failure"""
    predictor = FailingPredictor(settings=dummy_settings, predictor_type="failing")

    with pytest.raises(RuntimeError, match="Failed to load model"):
        predictor.load_model()


def test_unload_model(loaded_predictor, caplog):
    """Test model unloading"""
    assert loaded_predictor.model_loaded

    with caplog.at_level(logging.INFO):
        loaded_predictor.unload_model()

    assert not loaded_predictor.model_loaded
    assert loaded_predictor._model is None
    assert "Unloaded model" in caplog.text


def test_unload_model_not_loaded(dummy_predictor):
    """Test unloading when model not loaded"""
    dummy_predictor.unload_model()  # Should not raise any error
    assert not dummy_predictor.model_loaded


def test_predict_success(loaded_predictor):
    """Test successful prediction"""
    arr = np.ones((2, 2))
    pred = loaded_predictor.predict(arr)
    assert pred == 4.0


def test_predict_model_not_loaded(dummy_predictor):
    """Test prediction when model not loaded"""
    arr = np.ones((2, 2))
    with pytest.raises(RuntimeError, match="Model not loaded"):
        dummy_predictor.predict(arr)


def test_call_method(dummy_predictor):
    """Test __call__ method loads model and predicts"""
    arr = np.ones((2, 2))
    assert not dummy_predictor.model_loaded

    pred = dummy_predictor(arr)

    assert pred == 4.0
    assert dummy_predictor.model_loaded


def test_visualize(dummy_predictor):
    """Test visualization method"""
    arr = np.zeros((2, 2))
    vis = dummy_predictor.visualize(arr, predictions=None)
    assert np.array_equal(vis, arr)


def test_evaluate_with_prediction(dummy_predictor):
    """Test evaluation with pre-computed prediction"""
    metrics = dummy_predictor.evaluate(ground_truth=4.0, prediction=4.0)
    assert metrics["accuracy"] == 1.0
    assert metrics["mse"] == 0.0


def test_evaluate_with_input_data(loaded_predictor):
    """Test evaluation with input data"""
    arr = np.ones((2, 2))
    metrics = loaded_predictor.evaluate(ground_truth=4.0, input_data=arr)
    assert metrics["accuracy"] == 1.0
    assert metrics["mse"] == 0.0


def test_evaluate_no_prediction_no_input(dummy_predictor):
    """Test evaluation fails when neither prediction nor input data provided"""
    with pytest.raises(ValueError, match="Either 'prediction' or 'input_data' must be provided"):
        dummy_predictor.evaluate(ground_truth=4.0)


def test_predict_batch(loaded_predictor):
    """Test batch prediction"""
    arrs = [np.ones((2, 2)), np.zeros((2, 2))]
    preds = loaded_predictor.predict_batch(arrs, show_progress=False)
    assert sorted(preds) == [0.0, 4.0]


def test_predict_batch_loads_model(dummy_predictor):
    """Test batch prediction loads model if not loaded"""
    arrs = [np.ones((2, 2))]
    assert not dummy_predictor.model_loaded

    preds = dummy_predictor.predict_batch(arrs, show_progress=False)

    assert dummy_predictor.model_loaded
    assert preds == [4.0]


def test_evaluate_batch_with_predictions(dummy_predictor):
    """Test batch evaluation with pre-computed predictions"""
    predictions = [4.0, 0.0]
    ground_truths = [4.0, 0.0]

    metrics = dummy_predictor.evaluate_batch(
        ground_truth_batch=ground_truths,
        predictions_batch=predictions,
        show_progress=False,
    )

    assert metrics["accuracy_mean"] == 1.0
    assert metrics["mse_mean"] == 0.0
    assert "accuracy_std" in metrics
    assert "mse_std" in metrics


def test_evaluate_batch_with_input_data(loaded_predictor):
    """Test batch evaluation with input data"""
    arrs = [np.ones((2, 2)), np.zeros((2, 2))]
    ground_truths = [4.0, 0.0]

    metrics = loaded_predictor.evaluate_batch(
        ground_truth_batch=ground_truths,
        input_data_batch=arrs,
        show_progress=False,
    )

    assert metrics["accuracy_mean"] == 1.0
    assert metrics["mse_mean"] == 0.0


def test_evaluate_batch_no_predictions_no_input(dummy_predictor):
    """Test batch evaluation fails when neither predictions nor input data provided"""
    with pytest.raises(ValueError, match="Either 'predictions_batch' or 'input_data_batch' must be provided"):
        dummy_predictor.evaluate_batch(ground_truth_batch=[4.0])


def test_evaluate_batch_mismatched_lengths(dummy_predictor):
    """Test batch evaluation fails with mismatched batch lengths"""
    with pytest.raises(ValueError, match="Number of predictions .* must match number of ground truths"):
        dummy_predictor.evaluate_batch(ground_truth_batch=[4.0, 0.0], predictions_batch=[4.0], show_progress=False)


def test_aggregate_metrics_empty_list(dummy_predictor):
    """Test aggregating empty metrics list"""
    result = dummy_predictor._aggregate_metrics([])
    assert result == {}


def test_aggregate_metrics_with_empty_metrics(dummy_predictor, caplog):
    """Test aggregating metrics with some empty entries"""
    metrics_list = [{"accuracy": 1.0}, {}, {"accuracy": 0.0}]

    result = dummy_predictor._aggregate_metrics(metrics_list)

    assert result["accuracy_mean"] == 0.5
    assert result["count"] == 2


def test_aggregate_metrics_all_empty(dummy_predictor, caplog):
    """Test aggregating all empty metrics"""
    metrics_list = [{}, {}, {}]

    with caplog.at_level(logging.WARNING):
        result = dummy_predictor._aggregate_metrics(metrics_list)

    assert result == {}
    assert "No valid metrics found" in caplog.text


def test_aggregate_metrics_multiple_keys(dummy_predictor):
    """Test aggregating metrics with multiple keys"""
    metrics_list = [{"accuracy": 1.0, "precision": 0.8}, {"accuracy": 0.0, "precision": 0.6}]

    result = dummy_predictor._aggregate_metrics(metrics_list)

    assert result["accuracy_mean"] == pytest.approx(0.5)
    assert result["precision_mean"] == pytest.approx(0.7)
    assert result["accuracy_std"] == pytest.approx(0.5)
    assert result["precision_std"] == pytest.approx(0.1)


def test_finalize_evaluation_report(dummy_predictor):
    """Test finalizing evaluation report (default implementation)"""
    aggregated = {"accuracy_mean": 0.5}
    predictions = [4.0, 0.0]
    ground_truths = [4.0, 0.0]

    result = dummy_predictor._finalize_evaluation_report(aggregated, predictions, ground_truths)

    assert result == aggregated  # Default implementation returns unchanged


def test_model_context_manager_not_loaded(dummy_predictor):
    """Test model context manager with model not initially loaded"""
    assert not dummy_predictor.model_loaded

    with dummy_predictor.model_context() as predictor:
        assert predictor.model_loaded
        pred = predictor.predict(np.ones((2, 2)))
        assert pred == 4.0

    assert not dummy_predictor.model_loaded  # Should be unloaded after context


def test_model_context_manager_already_loaded(loaded_predictor):
    """Test model context manager with model already loaded"""
    assert loaded_predictor.model_loaded

    with loaded_predictor.model_context() as predictor:
        assert predictor.model_loaded
        pred = predictor.predict(np.ones((2, 2)))
        assert pred == 4.0

    assert loaded_predictor.model_loaded  # Should remain loaded


def test_context_manager_enter_exit(dummy_predictor):
    """Test context manager enter/exit methods"""
    assert not dummy_predictor.model_loaded

    with dummy_predictor as predictor:
        assert predictor.model_loaded
        pred = predictor.predict(np.ones((2, 2)))
        assert pred == 4.0

    assert dummy_predictor.model_loaded  # Model remains loaded (no auto-unload in __exit__)


def test_get_model_info(dummy_predictor):
    """Test getting model information"""
    info = dummy_predictor.get_model_info()

    assert info["predictor_type"] == "dummy"
    assert "dummy.pth" in info["model_path"]
    assert info["model_loaded"] is False
    assert info["config"]["repository_id"] == "dummy/repo"
    assert info["config"]["params"]["param1"] == "value1"
    dummy_predictor.config.model_dump.assert_called_once()


def test_get_model_info_loaded(loaded_predictor):
    """Test getting model information when loaded"""
    info = loaded_predictor.get_model_info()
    assert info["model_loaded"] is True


def test_calculate_metrics_parallel_success(dummy_predictor):
    """Test parallel metrics calculation success"""
    # Create real Future objects
    future1, future2 = Future(), Future()
    future1.set_result({"accuracy": 1.0})
    future2.set_result({"accuracy": 0.0})
    futures = [future1, future2]
    submit_calls = []

    class MockExecutor:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def submit(self, func, *args, **kwargs):
            submit_calls.append(args)
            return futures[len(submit_calls) - 1]

    with patch("culicidaelab.core.base_predictor.ThreadPoolExecutor", MockExecutor):
        predictions = [1, 2]
        ground_truths = [1, 2]
        result = dummy_predictor._calculate_metrics_parallel(
            predictions,
            ground_truths,
            num_workers=2,
            show_progress=False,
        )

    assert len(result) == 2
    accuracies = {r["accuracy"] for r in result}
    assert accuracies == {1.0, 0.0}


def test_calculate_metrics_parallel_with_exception(dummy_predictor, caplog):
    """Test parallel metrics calculation with exception"""
    future1, future2 = Future(), Future()
    future1.set_result({"accuracy": 1.0})
    future2.set_exception(Exception("Calculation failed"))
    futures = [future1, future2]
    submit_calls = []

    class MockExecutor:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def submit(self, func, *args, **kwargs):
            submit_calls.append(args)
            return futures[len(submit_calls) - 1]

    with patch("culicidaelab.core.base_predictor.ThreadPoolExecutor", MockExecutor):
        with caplog.at_level(logging.ERROR):
            result = dummy_predictor._calculate_metrics_parallel(
                [1, 2],
                [1, 2],
                num_workers=2,
                show_progress=False,
            )

    assert len(result) == 2
    assert any(r.get("accuracy") == 1.0 for r in result)
    assert any(r == {} for r in result)
    assert "Error calculating metrics for item" in caplog.text
    assert "Calculation failed" in caplog.text
