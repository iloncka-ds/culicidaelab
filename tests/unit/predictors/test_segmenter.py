import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# --- Mock necessary modules before they are imported by the test subject ---
try:
    from culicidaelab.core.config_models import PredictorConfig
except ImportError:
    PredictorConfig = type("PredictorConfig", (object,), {})
    sys.modules["culicidaelab.core.config_models"] = MagicMock(PredictorConfig=PredictorConfig)

# Import torch for creating device objects for testing
import torch

sys.modules["culicidaelab.core.settings"] = MagicMock()
sys.modules["culicidaelab.core.utils"] = MagicMock()
sys.modules["culicidaelab.predictors.model_weights_manager"] = MagicMock()

# This import must come AFTER the modules it depends on are mocked
from culicidaelab.predictors.segmenter import MosquitoSegmenter  # noqa: E402

# --- Fixtures ---


@pytest.fixture
def mock_settings():
    """Provides a consistent mock Settings object for tests."""
    settings = MagicMock(name="MockSettings")
    mock_predictor_config = MagicMock(spec=PredictorConfig)
    mock_predictor_config.device = "cpu"
    mock_viz_config = MagicMock(overlay_color="red", alpha=0.4)
    mock_predictor_config.visualization = mock_viz_config
    settings.get_config.return_value = mock_predictor_config
    return settings


@pytest.fixture(autouse=True)
def mock_common_dependencies(mocker):
    """Mocks common, non-model dependencies for all tests."""
    mocker.patch("culicidaelab.predictors.segmenter.str_to_bgr", return_value=[0, 0, 255])
    mock_mwm_instance = MagicMock()
    # Use a consistent, platform-independent path for the mock
    mock_mwm_instance.ensure_weights.return_value = Path("/mock/models/sam.pt")
    mocker.patch(
        "culicidaelab.predictors.segmenter.ModelWeightsManager",
        return_value=mock_mwm_instance,
    )


@pytest.fixture
def segmenter(mock_settings):
    """Provides a basic, un-loaded MosquitoSegmenter instance for each test."""
    return MosquitoSegmenter(settings=mock_settings, load_model=False)


@pytest.fixture
def loaded_segmenter(segmenter):
    """Provides a MosquitoSegmenter with a mocked and 'loaded' model."""
    mock_internal_predictor = MagicMock(name="mock_sam_instance")
    segmenter._model = mock_internal_predictor
    segmenter._model_loaded = True
    # Attach for easy access in tests
    segmenter.mocked_internal_predictor = mock_internal_predictor
    return segmenter


# --- Tests ---


def test_initialization(segmenter, mock_settings):
    assert not segmenter.model_loaded
    assert segmenter.config.device == "cpu"
    mock_settings.get_config.assert_called_once_with("predictors.segmenter")


def test_load_model(segmenter, mocker):
    mock_sam_constructor = mocker.patch("culicidaelab.predictors.segmenter.SAM")
    mock_sam_instance = MagicMock(name="sam_instance")
    mock_sam_constructor.return_value = mock_sam_instance
    segmenter.load_model()
    expected_path = segmenter.model_path.as_posix()
    mock_sam_constructor.assert_called_once_with(expected_path)
    mock_sam_instance.to.assert_called_once_with(torch.device("cpu"))
    assert segmenter.model_loaded is True


def test_load_model_failure(segmenter, mocker):
    mocker.patch("culicidaelab.predictors.segmenter.SAM", side_effect=ValueError("fail"))
    with pytest.raises(RuntimeError, match="Failed to load model"):
        segmenter.load_model()
    assert not segmenter.model_loaded


def test_predict_raises_if_model_fails(segmenter, mocker):
    mocker.patch.object(segmenter, "load_model", side_effect=RuntimeError("fail"))
    with pytest.raises(RuntimeError):
        segmenter.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_visualize_shape_mismatch(segmenter):
    input_image = np.zeros((10, 10, 3), dtype=np.uint8)
    prediction_mask = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(IndexError):
        segmenter.visualize(input_image, prediction_mask)


def test_evaluate_from_prediction_invalid_shapes(segmenter):
    pred_mask = np.ones((5, 5), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="Prediction and ground truth must have the same shape"):
        segmenter._evaluate_from_prediction(prediction=pred_mask, ground_truth=gt_mask)


def test_evaluate_handles_exception(loaded_segmenter, mocker):
    mocker.patch.object(loaded_segmenter, "predict", side_effect=Exception("fail"))
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(Exception, match="fail"):
        loaded_segmenter.evaluate(input_data=dummy_image, ground_truth=gt_mask)


def test_predict_triggers_load_model(segmenter, mocker):
    input_image = np.zeros((10, 10, 3), dtype=np.uint8)
    detection_boxes = [(1, 1, 8, 8, 0.9)]
    mock_internal_model = MagicMock(name="mock_sam_instance")
    mock_result = MagicMock(masks=MagicMock(data=np.zeros((1, 10, 10), dtype=bool)))
    mock_internal_model.return_value = [mock_result]
    mocker.patch.object(segmenter, "_load_model", side_effect=lambda: setattr(segmenter, "_model", mock_internal_model))
    segmenter.predict(input_image, detection_boxes=detection_boxes)
    segmenter._load_model.assert_called_once()
    mock_internal_model.assert_called_once()
    call_args, call_kwargs = mock_internal_model.call_args
    np.testing.assert_array_equal(call_args[0], input_image)
    expected_kwargs = {"bboxes": [(1, 1, 8, 8)], "verbose": False}
    assert call_kwargs == expected_kwargs


def test_predict_without_boxes(loaded_segmenter):
    input_image = np.zeros((100, 100, 3), dtype=np.uint8)
    result_mask = loaded_segmenter.predict(input_image)
    loaded_segmenter.mocked_internal_predictor.assert_not_called()
    assert result_mask.shape == (100, 100)
    assert not result_mask.any()


def test_predict_with_boxes(loaded_segmenter):
    H, W = 100, 100
    input_image = np.zeros((H, W, 3), dtype=np.uint8)
    detection_boxes = [(15, 15, 25, 25, 0.9), (40, 40, 50, 50, 0.8)]
    mock_result = MagicMock()
    mock_masks = np.zeros((2, H, W), dtype=bool)
    mock_masks[0, 15:25, 15:25] = True
    mock_masks[1, 40:50, 40:50] = True
    mock_result.masks = MagicMock(data=mock_masks)
    loaded_segmenter.mocked_internal_predictor.return_value = [mock_result]
    result_mask = loaded_segmenter.predict(input_image, detection_boxes=detection_boxes)
    expected_bboxes = [(15, 15, 25, 25), (40, 40, 50, 50)]
    loaded_segmenter.mocked_internal_predictor.assert_called_once()
    call_args, call_kwargs = loaded_segmenter.mocked_internal_predictor.call_args
    np.testing.assert_array_equal(call_args[0], input_image)
    assert call_kwargs == {"bboxes": expected_bboxes, "verbose": False}
    expected_mask = mock_masks.any(axis=0).astype(np.uint8)
    np.testing.assert_array_equal(result_mask, expected_mask)


def test_visualize(segmenter):
    input_image = np.zeros((10, 10, 3), dtype=np.uint8)
    prediction_mask = np.ones((10, 10), dtype=np.uint8)
    segmenter.visualize(input_image, prediction_mask)


@pytest.mark.parametrize(
    "name, pred_data, gt_data, exp_iou, exp_prec, exp_rec, exp_f1",
    [
        ("Perfect match", [[1, 1]], [[1, 1]], 1.0, 1.0, 1.0, 1.0),
        ("No overlap", [[1, 0]], [[0, 1]], 0.0, 0.0, 0.0, 0.0),
        ("Partial overlap", [[1, 1]], [[1, 0]], 0.5, 0.5, 1.0, 2 / 3),
        ("Pred empty", [[0, 0]], [[1, 1]], 0.0, 0.0, 0.0, 0.0),
        ("GT empty", [[1, 1]], [[0, 0]], 0.0, 0.0, 0.0, 0.0),
        ("Both empty", [[0, 0]], [[0, 0]], 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_evaluate_from_prediction(segmenter, name, pred_data, gt_data, exp_iou, exp_prec, exp_rec, exp_f1):
    pred_mask = np.array(pred_data, dtype=np.uint8)
    gt_mask = np.array(gt_data, dtype=np.uint8)
    metrics = segmenter._evaluate_from_prediction(prediction=pred_mask, ground_truth=gt_mask)
    assert metrics["iou"] == pytest.approx(exp_iou)
    assert metrics["precision"] == pytest.approx(exp_prec)
    assert metrics["recall"] == pytest.approx(exp_rec)
    assert metrics["f1"] == pytest.approx(exp_f1)


def test_evaluate_integration(loaded_segmenter, mocker):
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)
    pred_mask = np.zeros((10, 10), dtype=np.uint8)
    mocker.patch.object(loaded_segmenter, "predict", return_value=pred_mask)
    metrics = loaded_segmenter.evaluate(input_data=dummy_image, ground_truth=gt_mask)
    loaded_segmenter.predict.assert_called_once_with(dummy_image)
    assert metrics["iou"] == 0.0
