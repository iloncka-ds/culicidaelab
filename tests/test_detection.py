"""
Tests for the detection module.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from culicidaelab.detection import MosquitoDetector


@pytest.fixture
def mock_yolo():
    """Create a mock YOLO model."""
    with patch("culicidaelab.detection.YOLO") as mock:
        # Mock detection results
        mock_results = MagicMock()
        mock_box = MagicMock()
        mock_box.xyxy = [torch.tensor([[100, 100, 200, 200]])]
        mock_box.conf = [torch.tensor([0.95])]
        mock_results.boxes = [mock_box]
        mock_instance = mock.return_value
        mock_instance.return_value = [mock_results]
        yield mock


@pytest.fixture
def detector(mock_yolo):
    """Create a MosquitoDetector instance with mocked YOLO model."""
    return MosquitoDetector(model_path="mock_model.pt")


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.zeros((416, 416, 3), dtype=np.uint8)


def test_init_with_empty_model_path():
    """Test initialization with empty model path."""
    with pytest.raises(ValueError, match="model_path must be provided"):
        MosquitoDetector(model_path="")


def test_init_with_valid_model_path(mock_yolo):
    """Test initialization with valid model path."""
    detector = MosquitoDetector(model_path="valid_model.pt")
    assert detector.confidence_threshold == 0.5
    mock_yolo.assert_called_once_with("valid_model.pt")


def test_detect_with_file_path(detector):
    """Test detection with image file path."""
    detections = detector.detect("test_image.jpg")
    assert len(detections) == 1
    x, y, w, h, conf = detections[0]
    assert conf == 0.95
    assert w == 100  # x2 - x1
    assert h == 100  # y2 - y1


def test_detect_with_numpy_array(detector, sample_image):
    """Test detection with numpy array."""
    detections = detector.detect(sample_image)
    assert len(detections) == 1
    x, y, w, h, conf = detections[0]
    assert conf == 0.95


def test_visualize_detections_with_file_path(detector):
    """Test visualization with image file path."""
    with patch("cv2.imread") as mock_imread:
        mock_imread.return_value = np.zeros((416, 416, 3), dtype=np.uint8)
        detections = [(100, 100, 50, 50, 0.95)]
        result = detector.visualize_detections("test_image.jpg", detections)
        assert isinstance(result, np.ndarray)
        assert result.shape == (416, 416, 3)


def test_visualize_detections_with_numpy_array(detector, sample_image):
    """Test visualization with numpy array."""
    detections = [(100, 100, 50, 50, 0.95)]
    result = detector.visualize_detections(sample_image, detections)
    assert isinstance(result, np.ndarray)
    assert result.shape == (416, 416, 3)


def test_calculate_iou():
    """Test IoU calculation."""
    detector = MosquitoDetector(model_path="mock_model.pt")

    # Test complete overlap
    box1 = [100, 100, 50, 50]  # x, y, w, h
    box2 = [100, 100, 50, 50]
    iou = detector.calculate_iou(box1, box2)
    assert iou == 1.0

    # Test no overlap
    box2 = [200, 200, 50, 50]
    iou = detector.calculate_iou(box1, box2)
    assert iou == 0.0

    # Test partial overlap
    box2 = [125, 125, 50, 50]
    iou = detector.calculate_iou(box1, box2)
    assert 0 < iou < 1


def test_evaluate_empty_boxes():
    """Test evaluation with empty boxes."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    metrics = detector.evaluate([], [], iou_threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["ap"] == 0.0
    assert metrics["map"] == 0.0


def test_evaluate_perfect_detection():
    """Test evaluation with perfect detection."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    true_boxes = [[100, 100, 50, 50]]
    pred_boxes = [[100, 100, 50, 50, 1.0]]
    metrics = detector.evaluate(true_boxes, pred_boxes, iou_threshold=0.5)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["ap"] == 1.0


def test_evaluate_no_detection():
    """Test evaluation with no detections but ground truth exists."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    true_boxes = [[100, 100, 50, 50]]
    pred_boxes = []
    metrics = detector.evaluate(true_boxes, pred_boxes, iou_threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["ap"] == 0.0


@pytest.mark.integration
def test_train():
    """Test model training (marked as integration test)."""
    detector = MosquitoDetector(model_path="mock_model.pt")
    with patch.object(detector.model, "train") as mock_train:
        detector.train("data.yaml", epochs=1, batch_size=1)
        mock_train.assert_called_once_with(
            data="data.yaml",
            epochs=1,
            batch=1,
            imgsz=640,
            device="cuda:0",
        )
