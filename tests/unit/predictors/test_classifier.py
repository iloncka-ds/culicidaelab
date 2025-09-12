import pytest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

from culicidaelab.predictors.classifier import MosquitoClassifier
from culicidaelab.core.prediction_models import Classification, ClassificationPrediction


@pytest.fixture
def mock_backend():
    """Mocks the inference backend, conforming to the protocol."""
    backend = Mock()
    backend.is_loaded = False
    # The backend's predict method now returns a simple numpy array of probabilities
    backend.predict.return_value = np.array([0.9, 0.1])
    return backend


@pytest.fixture
def classifier(mock_settings, mock_backend):
    """Provides a MosquitoClassifier instance with a mocked backend."""
    # Patch the factory to inject our mock backend
    with patch("culicidaelab.predictors.classifier.create_backend", return_value=mock_backend):
        cls = MosquitoClassifier(settings=mock_settings, load_model=False)
        return cls


def test_init(classifier, mock_settings, mock_backend):
    """Test that the classifier initializes correctly."""
    assert classifier.settings is mock_settings
    assert classifier.predictor_type == "classifier"
    assert classifier.backend is mock_backend
    assert not mock_backend.load_model.called


def test_load_model(classifier, mock_backend):
    """Test that load_model delegates to the backend."""
    mock_backend.is_loaded = False
    classifier.load_model()
    # Assert that the predictor tells the backend to load itself
    mock_backend.load_model.assert_called_once_with("classifier")


def test_load_model_already_loaded(classifier, mock_backend):
    """Test that load_model doesn't do anything if the backend is already loaded."""
    mock_backend.is_loaded = True
    classifier.load_model()
    mock_backend.load_model.assert_not_called()


def test_unload_model(classifier, mock_backend):
    """Test that unload_model delegates to the backend."""
    mock_backend.is_loaded = True
    classifier.unload_model()
    mock_backend.unload_model.assert_called_once()


def test_predict_delegates_to_backend_and_parses_output(classifier, mock_backend):
    """Test that predict calls the backend and correctly parses the standardized output."""
    mock_backend.is_loaded = True
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Execute
    prediction = classifier.predict(dummy_image)

    # Assert backend was called
    mock_backend.predict.assert_called_once()

    # Assert the predictor correctly parsed the backend's numpy output
    assert isinstance(prediction, ClassificationPrediction)
    assert len(prediction.predictions) == 2
    assert prediction.predictions[0].species_name == "species_A"  # From mock_settings
    assert prediction.predictions[0].confidence == pytest.approx(0.9)
    assert prediction.predictions[1].species_name == "species_B"
    assert prediction.predictions[1].confidence == pytest.approx(0.1)


def test_predict_loads_model_if_not_loaded(classifier, mock_backend):
    """Test that predict automatically loads the model if needed."""
    mock_backend.is_loaded = False
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    classifier.predict(dummy_image)

    mock_backend.load_model.assert_called_once_with(predictor_type="classifier")
    mock_backend.predict.assert_called_once()


def test_predict_batch_serial_processing(classifier, mock_backend):
    """Test that predict_batch calls the backend and processes the results."""
    mock_backend.is_loaded = True
    # The backend returns a list of numpy arrays
    mock_backend.predict_batch.return_value = [
        np.array([0.9, 0.1]),
        np.array([0.2, 0.8]),
    ]

    images = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]
    predictions = classifier.predict_batch(images)

    mock_backend.predict_batch.assert_called_once()
    assert len(predictions) == 2
    assert isinstance(predictions[0], ClassificationPrediction)
    assert predictions[0].predictions[0].confidence == pytest.approx(0.9)
    assert predictions[1].predictions[0].confidence == pytest.approx(0.8)


def test_evaluate_from_prediction(classifier):
    """Test the internal evaluation logic (no change needed here, but good to keep)."""
    prediction = ClassificationPrediction(
        predictions=[
            Classification(species_name="Species A", confidence=0.9),
            Classification(species_name="Species B", confidence=0.1),
        ],
    )
    # The ground truth should be the mapped label if one exists
    metrics = classifier._evaluate_from_prediction(prediction, "species_A")
    print(classifier.labels_map)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["confidence"] == pytest.approx(0.9)


def test_visualize(classifier):
    """Test the visualization logic (no change needed, but good to keep)."""
    dummy_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    prediction = ClassificationPrediction(
        predictions=[
            Classification(species_name="species_A", confidence=0.9),
        ],
    )

    vis_img = classifier.visualize(dummy_image, prediction)

    assert isinstance(vis_img, np.ndarray)
    assert vis_img.shape[0] == 100
    assert vis_img.shape[1] > 100  # Panel is added
