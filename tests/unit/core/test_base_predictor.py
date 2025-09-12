import pytest
from unittest.mock import Mock, patch
import numpy as np

from culicidaelab.core.base_predictor import BasePredictor


# A concrete implementation of the abstract BasePredictor for testing
class ConcreteTestPredictor(BasePredictor):
    def predict(self, input_data, **kwargs):
        # This method is called by __call__
        return self.backend.predict(input_data, **kwargs)

    def _evaluate_from_prediction(self, prediction, ground_truth):
        # Simple evaluation for testing
        return {"accuracy": 1.0 if prediction == ground_truth else 0.0}

    def visualize(self, input_data, predictions, save_path=None):
        # Dummy visualization
        return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def mock_backend():
    """Mocks the InferenceBackendProtocol."""
    backend = Mock()
    backend.is_loaded = False
    backend.predict.return_value = "prediction_result"
    return backend


@pytest.fixture
def predictor(mock_settings, mock_backend):
    """Provides a ConcreteTestPredictor instance with a mocked backend."""
    return ConcreteTestPredictor(
        settings=mock_settings,
        predictor_type="classifier",  # Type doesn't matter much for these tests
        backend=mock_backend,
        load_model=False,
    )


def test_init(predictor, mock_backend):
    """Test that the BasePredictor initializes correctly."""
    assert predictor.backend is mock_backend
    assert not predictor.model_loaded
    mock_backend.load_model.assert_not_called()


def test_load_model_delegates_to_backend(predictor, mock_backend):
    """Test that load_model() calls the backend's load_model()."""
    mock_backend.is_loaded = False
    predictor.load_model()
    mock_backend.load_model.assert_called_once_with(predictor.predictor_type)


def test_load_model_does_nothing_if_already_loaded(predictor, mock_backend):
    """Test that load_model() is a no-op if the backend is already loaded."""
    mock_backend.is_loaded = True
    predictor.load_model()
    mock_backend.load_model.assert_not_called()


def test_unload_model_delegates_to_backend(predictor, mock_backend):
    """Test that unload_model() calls the backend's unload_model()."""
    mock_backend.is_loaded = True
    predictor.unload_model()
    mock_backend.unload_model.assert_called_once()


def test_call_method_loads_and_predicts(predictor, mock_backend):
    """Test that calling the predictor instance loads the model and predicts."""
    mock_backend.is_loaded = False
    result = predictor("input_data")
    mock_backend.load_model.assert_called_once_with(predictor.predictor_type)
    mock_backend.predict.assert_called_once_with("input_data")
    assert result == "prediction_result"


def test_model_context_manager(predictor, mock_backend):
    """Test the model_context context manager."""
    mock_backend.is_loaded = False
    with predictor.model_context() as p:
        assert p is predictor
        # Backend is loaded inside the context
        mock_backend.load_model.assert_called_once_with(predictor.predictor_type)
        # Let's say is_loaded is now true
        mock_backend.is_loaded = True
    # After exiting, unload_model should be called
    mock_backend.unload_model.assert_called_once()


def test_evaluate_with_prediction(predictor):
    """Test that evaluate() uses a pre-computed prediction."""
    with patch.object(predictor, "_evaluate_from_prediction", return_value={"accuracy": 1.0}) as mock_eval:
        metrics = predictor.evaluate(ground_truth="gt", prediction="pred")
        mock_eval.assert_called_once_with(prediction="pred", ground_truth="gt")
        assert metrics["accuracy"] == 1.0


def test_evaluate_with_input_data(predictor, mock_backend):
    """Test that evaluate() generates a prediction if not provided."""
    mock_backend.is_loaded = True
    with patch.object(predictor, "_evaluate_from_prediction", return_value={"accuracy": 1.0}) as mock_eval:
        metrics = predictor.evaluate(ground_truth="prediction_result", input_data="input")
        mock_backend.predict.assert_called_once_with("input")
        mock_eval.assert_called_once_with(prediction="prediction_result", ground_truth="prediction_result")
        assert metrics["accuracy"] == 1.0


def test_get_model_info(predictor, mock_backend):
    """Test that get_model_info returns correct data from the backend."""
    mock_backend.is_loaded = True
    info = predictor.get_model_info()
    assert info["predictor_type"] == "classifier"
    assert info["model_loaded"] is True
    assert "config" in info
