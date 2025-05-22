"""
Tests for the detection module.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from culicidaelab.detector import MosquitoDetector


@pytest.fixture
def mock_yolo():
    """Create a mock YOLO model."""
    with patch("culicidaelab.detection.YOLO") as mock:
        # Mock detection results
        mock_results = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = torch.tensor([[100.0, 100.0, 200.0, 200.0]])  # Shape: (1, 4)
        mock_box.conf = torch.tensor([0.95])  # Shape: (1,)
        mock_results.boxes = [mock_box]
        mock_instance = mock.return_value
        mock_instance.return_value = [mock_results]
        yield mock


@pytest.fixture
def detector(mock_yolo):
    """Create a MosquitoDetector instance with mocked YOLO model."""
    with patch("pathlib.Path.exists", return_value=True):  # Mock file existence
        return MosquitoDetector(model_path="mock_model.pt")


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.zeros((416, 416, 3), dtype=np.uint8)


def test_init_with_empty_model_path(mock_yolo):
    """Test initialization with empty model path."""
    with pytest.raises(ValueError, match="model_path must be provided"):
        MosquitoDetector(model_path="")


def test_init_with_valid_model_path(mock_yolo):
    """Test initialization with valid model path."""
    detector = MosquitoDetector(model_path="valid_model.pt")
    assert detector.confidence_threshold == 0.5
    mock_yolo.assert_called_once_with("valid_model.pt")


def test_detect_with_file_path(detector, mock_yolo):
    """Test detection with image file path."""
    detections = detector.detect("test_image.jpg")
    assert len(detections) == 1
    x, y, w, h, conf = detections[0]
    assert abs(conf - 0.95) < 1e-6
    assert w == 100  # x2 - x1
    assert h == 100  # y2 - y1


def test_detect_with_numpy_array(detector, sample_image, mock_yolo):
    """Test detection with numpy array."""
    detections = detector.detect(sample_image)
    assert len(detections) == 1
    x, y, w, h, conf = detections[0]
    assert abs(conf - 0.95) < 1e-6


def test_visualize_detections_with_file_path(detector, mock_yolo):
    """Test visualization with image file path."""
    with patch("cv2.imread") as mock_imread:
        mock_imread.return_value = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = [(100, 100, 50, 50, 0.95)]
        result = detector.visualize_detections("test_image.jpg", detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == (416, 416, 3)


def test_visualize_detections_with_numpy_array(detector, sample_image, mock_yolo):
    """Test visualization with numpy array."""
    detections = [(100, 100, 50, 50, 0.95)]
    result = detector.visualize_detections(sample_image, detections)
    assert isinstance(result, np.ndarray)
    assert result.shape == (416, 416, 3)


def test_calculate_iou(mock_yolo):
    """Test IoU calculation."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    box1 = [100, 100, 50, 50]  # x, y, w, h
    box2 = [125, 125, 50, 50]  # x, y, w, h
    iou = detector.calculate_iou(box1, box2)
    assert abs(iou - 0.14285714285714285) < 1e-6  # Correct IoU value for these boxes


def test_evaluate_empty_boxes(mock_yolo):
    """Test evaluation with empty boxes."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    metrics = detector.evaluate([], [])
    assert metrics["precision"] == 0.0  # No predictions, so precision is 0
    assert metrics["recall"] == 0.0  # No true boxes, so recall is 0
    assert metrics["f1"] == 0.0  # Both precision and recall are 0


def test_evaluate_perfect_detection(mock_yolo):
    """Test evaluation with perfect detection."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    true_boxes = [[100, 100, 50, 50]]  # x, y, w, h
    pred_boxes = [[100, 100, 50, 50, 1.0]]  # x, y, w, h, conf
    metrics = detector.evaluate(true_boxes, pred_boxes)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_no_detection(mock_yolo):
    """Test evaluation with no detections but ground truth exists."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    true_boxes = [[100, 100, 50, 50]]  # x, y, w, h
    pred_boxes = []  # No predictions
    metrics = detector.evaluate(true_boxes, pred_boxes)
    assert metrics["precision"] == 0.0  # No predictions, so precision is 0
    assert metrics["recall"] == 0.0  # No matches, so recall is 0
    assert metrics["f1"] == 0.0  # Both precision and recall are 0


@pytest.mark.integration
def test_train(mock_yolo):
    """Test model training (marked as integration test)."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    # Mock training data
    train_data = "path/to/train_data"
    val_data = "path/to/val_data"
    epochs = 1
    detector.train(train_data, val_data, epochs)
