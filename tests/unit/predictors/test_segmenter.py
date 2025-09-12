import pytest
import numpy as np
from unittest.mock import Mock, patch
from PIL import Image

from culicidaelab.predictors.segmenter import MosquitoSegmenter
from culicidaelab.core.prediction_models import SegmentationPrediction


# 1. Mock Backend Fixture
@pytest.fixture
def mock_backend():
    """Mocks the inference backend for a segmenter."""
    backend = Mock()
    backend.is_loaded = False
    # The backend's predict method now returns a simple numpy array (the mask)
    # Shape: (H, W)
    backend.predict.return_value = np.ones((100, 100), dtype=np.uint8)
    return backend


# 2. Patched Segmenter Fixture
@pytest.fixture
def segmenter(mock_settings, mock_backend):
    """Provides a MosquitoSegmenter instance with a mocked backend."""
    with patch("culicidaelab.predictors.segmenter.create_backend", return_value=mock_backend):
        seg = MosquitoSegmenter(settings=mock_settings, load_model=False)
        return seg


# 3. New Test Cases


def test_init(segmenter, mock_settings, mock_backend):
    """Test that the segmenter initializes correctly."""
    assert segmenter.settings is mock_settings
    assert segmenter.predictor_type == "segmenter"
    assert segmenter.backend is mock_backend
    assert not mock_backend.load_model.called


def test_load_model(segmenter, mock_backend):
    """Test that load_model delegates to the backend."""
    mock_backend.is_loaded = False
    segmenter.load_model()
    mock_backend.load_model.assert_called_once_with("segmenter")


def test_predict_delegates_to_backend_and_parses_output(segmenter, mock_backend):
    """Test that predict calls the backend and correctly wraps the standardized output."""
    mock_backend.is_loaded = True
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = [[10, 10, 90, 90]]

    prediction = segmenter.predict(dummy_image, detection_boxes=boxes)

    # Assert backend was called with correct prompts
    mock_backend.predict.assert_called_once()
    call_kwargs = mock_backend.predict.call_args.kwargs
    assert call_kwargs["bboxes"] == boxes

    # Assert the predictor correctly wrapped the backend's numpy mask
    assert isinstance(prediction, SegmentationPrediction)
    assert prediction.mask.shape == (100, 100)
    assert prediction.pixel_count == np.sum(mock_backend.predict.return_value)


def test_predict_no_prompts_returns_empty_mask(segmenter, mock_backend):
    """Test that predict returns an empty mask if no prompts are given."""
    mock_backend.is_loaded = True
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

    prediction = segmenter.predict(dummy_image)

    # The backend should NOT be called if there are no prompts
    mock_backend.predict.assert_not_called()
    assert isinstance(prediction, SegmentationPrediction)
    assert prediction.mask.shape == (100, 100)
    assert prediction.pixel_count == 0


def test_predict_batch_serial_processing(segmenter, mock_backend):
    """Test that predict_batch calls the backend and processes the results."""
    mock_backend.is_loaded = True
    # The backend returns a list of numpy arrays (masks)
    mask1 = np.ones((100, 100), dtype=np.uint8)
    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mock_backend.predict_batch.return_value = [mask1, mask2]

    images = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]
    boxes = [[[10, 10, 90, 90]], []]

    predictions = segmenter.predict_batch(images, detection_boxes_batch=boxes)

    mock_backend.predict_batch.assert_called_once()
    assert len(predictions) == 2
    assert predictions[0].pixel_count == 10000
    assert predictions[1].pixel_count == 0


@pytest.mark.parametrize(
    "pred_data, gt_data, expected_iou",
    [
        (np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]]), 1.0),  # Perfect
        (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), 0.0),  # No overlap
        (np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]), 0.3333),  # Partial
    ],
)
def test_evaluate_from_prediction(segmenter, pred_data, gt_data, expected_iou):
    """Test the internal evaluation logic for IoU calculation."""
    prediction = SegmentationPrediction(mask=pred_data, pixel_count=np.sum(pred_data))
    metrics = segmenter._evaluate_from_prediction(prediction, gt_data)
    assert metrics["iou"] == pytest.approx(expected_iou, abs=1e-4)


def test_visualize(segmenter):
    """Test the visualization logic."""
    dummy_image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    prediction = SegmentationPrediction(mask=np.ones((100, 100), dtype=np.uint8), pixel_count=10000)

    vis_img = segmenter.visualize(dummy_image, prediction)

    assert isinstance(vis_img, np.ndarray)
    assert vis_img.shape == (100, 100, 3)
