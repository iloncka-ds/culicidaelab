import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch

# Import the Pydantic model we need to instantiate
from culicidaelab.core.config_models import PredictorConfig
from culicidaelab.predictors.classifier import MosquitoClassifier
from culicidaelab.core.base_predictor import BasePredictor

# --- Fixtures ---


@pytest.fixture
def mock_predictor_config():
    """
    Provides a real, valid instance of the PredictorConfig Pydantic model.
    This is the key fix to satisfy the `isinstance` check.
    """
    # Instantiate the real Pydantic model with data for the test
    return PredictorConfig(
        _target_="some.dummy.class.path",  # Fulfill required field
        model_path="dummy/path/model.pkl",  # Fulfill required field
        params={"top_k": 3},
        # Add 'visualization' as an extra field, allowed by `extra="allow"` in the model
        visualization={
            "font_scale": 0.7,
            "text_thickness": 2,
            "text_color": (0, 255, 0),
        },
        model_arch="efficientnet_b0",
    )


@pytest.fixture
def mock_species_map():
    """Provides the species name-to-index mapping."""
    return {0: "species1", 1: "species2", 2: "species3"}


@pytest.fixture
def mock_settings(tmp_path, mock_predictor_config, mock_species_map):
    """Mocks the main Settings object passed to the classifier."""
    settings = Mock()

    # This side_effect logic is still correct. It will now return our
    # real PredictorConfig instance when called with the correct path.
    def get_config_side_effect(path: str, default=None):
        if path == "predictors.classifier":
            return mock_predictor_config
        return default

    settings.get_config.side_effect = get_config_side_effect

    # Mock the species_config attribute
    species_config_mock = Mock()
    species_config_mock.species_map = mock_species_map
    inverse_map = {v: k for k, v in mock_species_map.items()}
    species_config_mock.get_index_by_species.side_effect = lambda name: inverse_map.get(name)
    settings.species_config = species_config_mock

    # Mock resource directory properties
    settings.dataset_dir = tmp_path
    settings.model_dir = tmp_path

    return settings


@pytest.fixture
def classifier(mock_settings, tmp_path):
    """Provides an instance of MosquitoClassifier with mocked dependencies."""
    with patch("culicidaelab.core.base_predictor.ModelWeightsManager") as mock_manager:
        mock_manager.return_value.ensure_weights.return_value = tmp_path / "dummy.pkl"
        # This instantiation will now succeed
        clf = MosquitoClassifier(settings=mock_settings, load_model=False)
        clf.learner = Mock()
        yield clf


# --- Test Cases (No changes needed in the test logic itself) ---


def test_initialization(classifier, mock_settings, mock_species_map):
    """Test that the classifier initializes correctly via the BasePredictor."""
    assert classifier.predictor_type == "classifier"
    assert isinstance(classifier, BasePredictor)
    mock_settings.get_config.assert_called_with("predictors.classifier")
    assert classifier.arch == "efficientnet_b0"
    assert classifier.num_classes == len(mock_species_map)
    assert classifier.species_map == mock_species_map


def test_load_model_success(classifier):
    """Test successful model loading via the public load_model() method."""
    with patch("culicidaelab.predictors.classifier.load_learner") as mock_load_learner:
        mock_learner_instance = Mock()
        mock_load_learner.return_value = mock_learner_instance
        classifier.load_model()
        mock_load_learner.assert_called_once_with(classifier.model_path)
        assert classifier.model_loaded is True
        assert classifier.learner == mock_learner_instance


def test_load_model_failure(classifier):
    """Test that model loading failure raises a RuntimeError."""
    with patch("culicidaelab.predictors.classifier.load_learner", side_effect=Exception("Load failed")):
        with pytest.raises(RuntimeError, match="Failed to load model for classifier"):
            classifier.load_model()
        assert classifier.model_loaded is False


def test_predict(classifier):
    """Test the predict method with a mocked loaded model."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_probs = torch.tensor([0.1, 0.7, 0.2])
    classifier.learner.predict.return_value = ("species2", 1, mock_probs)
    classifier._model_loaded = True
    results = classifier.predict(dummy_image)
    classifier.learner.predict.assert_called_once()
    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] == ("species2", pytest.approx(0.7))


def test_predict_model_not_loaded(classifier):
    """Test that predict raises a RuntimeError if the model is not loaded."""
    classifier._model_loaded = False
    with pytest.raises(RuntimeError, match="Model is not loaded"):
        classifier.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_visualize(classifier):
    """Test the visualize method."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    predictions = [("species1", 0.9), ("species2", 0.08), ("species3", 0.02)]
    with patch("cv2.putText") as mock_putText:
        vis_img = classifier.visualize(dummy_image, predictions)
        assert isinstance(vis_img, np.ndarray)
        assert vis_img.shape == dummy_image.shape
        assert mock_putText.call_count == 3


def test_evaluate_single_item_correct(classifier):
    """Test the public BasePredictor.evaluate method for a correct prediction."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    ground_truth = "species1"
    prediction_result = [("species1", 0.9), ("species2", 0.1)]
    with patch.object(classifier, "predict", return_value=prediction_result) as mock_predict:
        metrics = classifier.evaluate(input_data=dummy_image, ground_truth=ground_truth)
        mock_predict.assert_called_once_with(dummy_image)
        assert metrics["accuracy"] == 1.0
        assert metrics["confidence"] == 0.9


def test_evaluate_single_item_incorrect_but_in_top5(classifier):
    """Test BasePredictor.evaluate for an incorrect prediction that is in top 5."""
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    ground_truth = "species2"
    prediction_result = [("species1", 0.8), ("species2", 0.15), ("species3", 0.05)]
    with patch.object(classifier, "predict", return_value=prediction_result):
        metrics = classifier.evaluate(input_data=dummy_image, ground_truth=ground_truth)
        assert metrics["accuracy"] == 0.0
        assert metrics["top_5_correct"] == 1.0


def test_evaluate_batch(classifier):
    """Test the full BasePredictor.evaluate_batch flow."""
    dummy_images = [np.zeros((10, 10, 3), dtype=np.uint8)] * 2
    ground_truths = ["species1", "species2"]
    predictions = [
        [("species1", 0.9), ("species2", 0.08), ("species3", 0.02)],
        [("species1", 0.8), ("species2", 0.15), ("species3", 0.05)],
    ]
    with patch.object(classifier, "predict_batch", return_value=predictions):
        report = classifier.evaluate_batch(
            input_data_batch=dummy_images,
            ground_truth_batch=ground_truths,
            num_workers=1,
        )
        assert report["accuracy_mean"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2
        assert report["confidence_mean"] == pytest.approx(0.85)  # (0.9 + 0.8) / 2
        assert report["top_5_correct_mean"] == pytest.approx(1.0)  # (1.0 + 1.0) / 2
        assert "confusion_matrix" in report
        expected_cm = [[1, 0, 0], [1, 0, 0], [0, 0, 0]]
        assert report["confusion_matrix"] == expected_cm


def test_get_species_names_and_index(classifier):
    """Test the species utility methods."""
    names = classifier.get_species_names()
    assert names == ["species1", "species2", "species3"]
    assert classifier.get_class_index("species2") == 1
    assert classifier.get_class_index("non_existent_species") is None
