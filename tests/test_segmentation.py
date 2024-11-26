"""
Tests for the segmentation module.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from culicidaelab.segmentation import MosquitoSegmenter


@pytest.fixture
def mock_sam():
    """Create a mock SAM model."""
    with patch("culicidaelab.segmentation.sam_model_registry") as mock_registry:
        mock_model = MagicMock()
        mock_registry.__getitem__.return_value = lambda checkpoint: mock_model
        yield mock_registry


@pytest.fixture
def mock_predictor():
    """Create a mock SAM predictor."""
    with patch("culicidaelab.segmentation.SamPredictor") as mock:
        mock_instance = mock.return_value

        # Mock predict method to return empty mask for empty boxes
        def predict_side_effect(**kwargs):
            if "box" in kwargs:
                return [np.ones((416, 416), dtype=bool)], [0.95], [1]
            return [], [], []

        mock_instance.predict.side_effect = predict_side_effect
        # Mock generate method to return empty list for empty detection boxes
        mock_instance.generate.return_value = []
        yield mock_instance


@pytest.fixture
def segmenter(mock_sam, mock_predictor):
    """Create a MosquitoSegmenter instance with mocked components."""
    with patch("torch.cuda.is_available", return_value=False):
        return MosquitoSegmenter(checkpoint_path="mock_checkpoint.pth")


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.zeros((416, 416, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask for testing."""
    mask = np.zeros((416, 416), dtype=np.uint8)
    mask[100:200, 100:200] = 1  # Create a square mask
    return mask


def test_init_with_empty_checkpoint():
    """Test initialization with empty checkpoint path."""
    with pytest.raises(ValueError, match="checkpoint_path must be provided"):
        MosquitoSegmenter(checkpoint_path="")


def test_init_with_valid_checkpoint(mock_sam):
    """Test initialization with valid checkpoint path."""
    with patch("torch.cuda.is_available", return_value=False):
        segmenter = MosquitoSegmenter(checkpoint_path="valid_checkpoint.pth")
        assert isinstance(segmenter.device, torch.device)
        assert str(segmenter.device) == "cpu"


def test_segment_with_file_path(segmenter, mock_predictor):
    """Test segmentation with image file path."""
    with patch("cv2.imread") as mock_imread, patch("cv2.cvtColor") as mock_cvtcolor:
        mock_imread.return_value = np.zeros((416, 416, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((416, 416, 3), dtype=np.uint8)

        mask = segmenter.segment("test_image.jpg")
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (416, 416)
        assert mask.dtype == np.uint8


def test_segment_with_numpy_array(segmenter, mock_predictor, sample_image):
    """Test segmentation with numpy array input."""
    mask = segmenter.segment(sample_image)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (416, 416)
    assert mask.dtype == np.uint8


def test_segment_with_detection_boxes(segmenter, mock_predictor, sample_image):
    """Test segmentation with detection boxes."""
    detection_boxes = [(100, 100, 50, 50, 0.95)]
    mask = segmenter.segment(sample_image, detection_boxes)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (416, 416)
    assert mask.dtype == np.uint8


def test_segment_with_empty_detection_boxes(segmenter, mock_predictor, sample_image):
    """Test segmentation with empty detection boxes."""
    mask = segmenter.segment(sample_image, [])
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (416, 416)
    assert mask.dtype == np.uint8
    assert not mask.any()  # Should be all zeros


def test_apply_mask_with_file_path(segmenter, sample_mask):
    """Test mask application with image file path."""
    with patch("cv2.imread") as mock_imread, patch("cv2.cvtColor") as mock_cvtcolor:
        mock_imread.return_value = np.zeros((416, 416, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((416, 416, 3), dtype=np.uint8)

        result = segmenter.apply_mask("test_image.jpg", sample_mask)
        assert isinstance(result, np.ndarray)
        assert result.shape == (416, 416, 3)
        assert result.dtype == np.uint8


def test_apply_mask_with_numpy_array(segmenter, sample_image, sample_mask):
    """Test mask application with numpy array input."""
    result = segmenter.apply_mask(sample_image, sample_mask)
    assert isinstance(result, np.ndarray)
    assert result.shape == (416, 416, 3)
    assert result.dtype == np.uint8


def test_evaluate_empty_masks(segmenter):
    """Test evaluation with empty masks."""
    metrics = segmenter.evaluate([], [], iou_threshold=0.5)
    assert metrics["mean_iou"] == 0.0
    assert metrics["mean_dice"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_perfect_match(segmenter):
    """Test evaluation with perfectly matching masks."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    metrics = segmenter.evaluate([mask], [mask], iou_threshold=0.5)
    assert metrics["mean_iou"] == 1.0
    assert metrics["mean_dice"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_evaluate_no_overlap(segmenter):
    """Test evaluation with non-overlapping masks."""
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:5, 0:5] = True
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[5:10, 5:10] = True
    metrics = segmenter.evaluate([mask1], [mask2], iou_threshold=0.5)
    assert metrics["mean_iou"] == 0.0
    assert metrics["mean_dice"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_evaluate_partial_overlap(segmenter):
    """Test evaluation with partially overlapping masks."""
    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[0:6, 0:6] = True
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[4:10, 4:10] = True
    metrics = segmenter.evaluate([mask1], [mask2], iou_threshold=0.5)
    assert 0 < metrics["mean_iou"] < 1
    assert 0 < metrics["mean_dice"] < 1
    assert 0 < metrics["precision"] < 1
    assert 0 < metrics["recall"] < 1
    assert 0 < metrics["f1"] < 1


def test_evaluate_mismatched_shapes(segmenter):
    """Test evaluation with mismatched mask shapes."""
    mask1 = np.zeros((10, 10), dtype=bool)
    mask2 = np.zeros((20, 20), dtype=bool)
    with pytest.raises(ValueError, match="Mask shapes must match"):
        segmenter.evaluate([mask1], [mask2])


def test_evaluate_mismatched_counts(segmenter):
    """Test evaluation with mismatched mask counts."""
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="Number of true and predicted masks must match"):
        segmenter.evaluate([mask], [mask, mask])
