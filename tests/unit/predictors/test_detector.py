import pytest
import numpy as np
from unittest.mock import Mock, patch
from PIL import Image

from culicidaelab.predictors.detector import MosquitoDetector
from culicidaelab.core.prediction_models import BoundingBox, Detection, DetectionPrediction


# 1. Mock Backend Fixture
@pytest.fixture
def mock_backend():
    """Mocks the inference backend for a detector."""
    backend = Mock()
    backend.is_loaded = False
    # The backend's predict method now returns a simple numpy array
    # Shape: (N, 5) -> [x1, y1, x2, y2, confidence]
    backend.predict.return_value = np.array(
        [
            [10.0, 20.0, 110.0, 120.0, 0.9],
            [30.0, 40.0, 130.0, 140.0, 0.8],
        ],
    )
    return backend


# 2. Patched Detector Fixture
@pytest.fixture
def detector(mock_settings, mock_backend):
    """Provides a MosquitoDetector instance with a mocked backend."""
    with patch("culicidaelab.predictors.detector.create_backend", return_value=mock_backend):
        det = MosquitoDetector(settings=mock_settings, load_model=False)
        return det


# 3. New Test Cases


def test_init(detector, mock_settings, mock_backend):
    """Test that the detector initializes correctly."""
    assert detector.settings is mock_settings
    assert detector.predictor_type == "detector"
    assert detector.backend is mock_backend
    assert detector.confidence_threshold == 0.5  # Default from conftest
    assert not mock_backend.load_model.called


def test_load_model(detector, mock_backend):
    """Test that load_model delegates to the backend."""
    mock_backend.is_loaded = False
    detector.load_model()
    mock_backend.load_model.assert_called_once_with()


def test_predict_delegates_to_backend_and_parses_output(detector, mock_backend):
    """Test that predict calls the backend and correctly parses the standardized output."""
    mock_backend.is_loaded = True
    dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)

    prediction = detector.predict(dummy_image)

    mock_backend.predict.assert_called_once()
    assert isinstance(prediction, DetectionPrediction)
    assert len(prediction.detections) == 2

    # Check first detection
    assert prediction.detections[0].box.x1 == pytest.approx(10.0)
    assert prediction.detections[0].confidence == pytest.approx(0.9)

    # Check second detection
    assert prediction.detections[1].box.x2 == pytest.approx(130.0)
    assert prediction.detections[1].confidence == pytest.approx(0.8)


def test_predict_batch_uses_backend_batching(detector, mock_backend):
    """Test that predict_batch calls the backend once for the whole batch."""
    mock_backend.is_loaded = True
    dummy_images = [np.zeros((200, 200, 3))] * 2

    # The YOLO backend is efficient and takes the whole batch in one go.
    # It returns a list of standardized numpy arrays.
    mock_backend.predict_batch.return_value = [
        np.array([[10.0, 20.0, 110.0, 120.0, 0.9]]),
        np.array([[30.0, 40.0, 130.0, 140.0, 0.8]]),
    ]

    predictions = detector.predict_batch(dummy_images)

    mock_backend.predict_batch.assert_called_once()
    assert len(predictions) == 2
    assert isinstance(predictions[0], DetectionPrediction)
    assert len(predictions[0].detections) == 1
    assert len(predictions[1].detections) == 1
    assert predictions[1].detections[0].confidence == pytest.approx(0.8)


@pytest.mark.parametrize(
    "gt_boxes, pred_boxes, expected_ap",
    [
        # Perfect match
        ([(10, 10, 50, 50)], [Detection(box=BoundingBox(x1=10, y1=10, x2=50, y2=50), confidence=0.9)], 1.0),
        # No predictions
        ([(10, 10, 50, 50)], [], 0.0),
        # No ground truth
        ([], [Detection(box=BoundingBox(x1=10, y1=10, x2=50, y2=50), confidence=0.9)], 0.0),
        # No GT, no predictions
        ([], [], 1.0),
    ],
)
def test_evaluate_from_prediction(detector, gt_boxes, pred_boxes, expected_ap):
    """Test the internal evaluation logic with various scenarios."""
    # Set a consistent IoU threshold for the test
    detector.iou_threshold = 0.5
    prediction = DetectionPrediction(detections=pred_boxes)
    metrics = detector._evaluate_from_prediction(prediction, gt_boxes)
    assert metrics["ap"] == pytest.approx(expected_ap)


def test_visualize(detector):
    """Test the visualization logic."""
    dummy_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    prediction = DetectionPrediction(
        detections=[
            Detection(box=BoundingBox(x1=10, y1=10, x2=90, y2=90), confidence=0.9),
        ],
    )

    vis_img = detector.visualize(dummy_image, prediction)

    assert isinstance(vis_img, np.ndarray)
    assert vis_img.shape == (100, 100, 3)
