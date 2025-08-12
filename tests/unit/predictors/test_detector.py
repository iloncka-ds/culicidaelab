import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch

from culicidaelab.core.config_models import PredictorConfig
from culicidaelab.predictors.detector import MosquitoDetector


@pytest.fixture
def mock_predictor_config():
    """Provides a valid PredictorConfig instance for the detector."""
    return PredictorConfig(
        target="some.dummy.detector.class",
        model_path="dummy/path/yolo.pt",
        provider="mock_yolo_provider",
        confidence=0.25,
        params={
            "iou_threshold": 0.45,
            "max_detections": 100,
        },
        visualization={
            "box_color": "green",
            "text_color": "white",
            "font_scale": 0.5,
            "thickness": 2,
        },
        model_config_path="dummy/path/config.yaml",
        model_config_filename="config.yaml",
    )


@pytest.fixture
def mock_settings(mock_predictor_config):
    """Mocks the main Settings object."""
    settings = Mock()

    def get_config_side_effect(path: str, default=None):
        if path == "predictors.detector":
            return mock_predictor_config
        if path == "providers.mock_yolo_provider":
            return Mock(target="culicidaelab.core.providers.local.LocalProvider")
        return default

    settings.get_config.side_effect = get_config_side_effect
    return settings


@pytest.fixture
def mock_weights_manager(tmp_path):
    """Provides a mocked ModelWeightsManager."""
    manager = Mock()
    manager.ensure_weights.return_value = tmp_path / "dummy_yolo.pt"
    return manager


@pytest.fixture
def detector(mock_settings):
    """Provides a MosquitoDetector instance with mocked dependencies."""
    with patch("culicidaelab.predictors.detector.YOLO") as _:
        with patch(
            "culicidaelab.predictors.detector.ModelWeightsManager",
        ) as MockWeightsManager:
            instance = MockWeightsManager.return_value
            instance.ensure_weights.return_value = "dummy/path/yolo.pt"
            det = MosquitoDetector(settings=mock_settings, load_model=False)
            det._model = Mock()
            det._model.return_value = []
            det._model_loaded = True
            return det


def test_detector_initialization(detector, mock_settings):
    """Test that the detector initializes correctly."""
    assert detector.predictor_type == "detector"
    mock_settings.get_config.assert_called_with("predictors.detector")
    assert detector.confidence_threshold == 0.25
    assert detector.iou_threshold == 0.45


def test_predict_single_image(detector):
    """Test the predict method on a single image."""
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)

    mock_result = Mock()
    mock_box = Mock()
    mock_box.xyxy = torch.tensor([[10.0, 20.0, 110.0, 120.0]])
    mock_box.conf = torch.tensor([0.9])
    mock_result.boxes = [mock_box]
    detector._model.return_value = [mock_result]

    predictions = detector.predict(dummy_image)

    detector._model.assert_called_once()
    assert isinstance(predictions, list)
    assert len(predictions) == 1
    cx, cy, w, h, conf = predictions[0]
    assert cx == pytest.approx(60)
    assert cy == pytest.approx(70)
    assert w == pytest.approx(100)
    assert h == pytest.approx(100)
    assert conf == pytest.approx(0.9)


@pytest.mark.parametrize(
    "gt, pred, expected_ap",
    [
        ([(60, 70, 100, 100)], [(60, 70, 100, 100, 0.9)], 1.0),
        ([(60, 70, 100, 100)], [], 0.0),
        ([], [(60, 70, 100, 100, 0.9)], 0.0),
        ([], [], 1.0),
    ],
)
def test_evaluate_from_prediction(detector, gt, pred, expected_ap):
    """Test the core evaluation logic with different scenarios."""
    metrics = detector._evaluate_from_prediction(pred, gt)
    assert metrics["ap"] == pytest.approx(expected_ap)


def test_predict_batch_efficiently(detector):
    """Test the batch prediction method."""
    dummy_images = [np.zeros((640, 640, 3), dtype=np.uint8)] * 2

    mock_result1 = Mock()
    mock_box1 = Mock()
    mock_box1.xyxy = torch.tensor([[10.0, 20.0, 110.0, 120.0]])
    mock_box1.conf = torch.tensor([0.9])
    mock_result1.boxes = [mock_box1]

    mock_result2 = Mock()
    mock_box2 = Mock()
    mock_box2.xyxy = torch.tensor([[30.0, 40.0, 130.0, 140.0]])
    mock_box2.conf = torch.tensor([0.8])
    mock_result2.boxes = [mock_box2]

    detector._model.return_value = [mock_result1, mock_result2]

    predictions_batch = detector.predict_batch(dummy_images, show_progress=False)

    detector._model.assert_called_once()
    assert isinstance(predictions_batch, list)
    assert len(predictions_batch) == 2
    assert len(predictions_batch[0]) == 1
    assert len(predictions_batch[1]) == 1
    assert predictions_batch[1][0][-1] == pytest.approx(0.8)
