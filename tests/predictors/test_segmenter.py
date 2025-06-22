import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# --- Import the class to be tested ---
from culicidaelab.predictors.segmenter import MosquitoSegmenter

# --- Mocking external and project-internal dependencies ---

# cv2 is used in the test file through mocks
try:
    import cv2  # noqa: F401
except ImportError:
    cv2_mock = MagicMock(name="global_cv2_mock")
    sys.modules["cv2"] = cv2_mock

# sam2 imports are used in the test file through mocks
try:
    from sam2.build_sam import build_sam2  # noqa: F401
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: F401
except ImportError:
    sys.modules["sam2"] = MagicMock()
    sys.modules["sam2.build_sam"] = MagicMock()
    sys.modules["sam2.sam2_image_predictor"] = MagicMock()

# Mock Pydantic config model if not available
try:
    from culicidaelab.core.config_models import PredictorConfig
except ImportError:
    PredictorConfig = type("PredictorConfig", (object,), {})  # type: ignore[assignment, misc]
    if "culicidaelab.core.config_models" not in sys.modules:
        sys.modules["culicidaelab.core.config_models"] = MagicMock(PredictorConfig=PredictorConfig)

sys.modules["culicidaelab.core.settings"] = MagicMock()
sys.modules["culicidaelab.core.utils"] = MagicMock()
sys.modules["culicidaelab.core.model_weights_manager"] = MagicMock()

# --- Test Fixtures ---


@pytest.fixture
def mock_settings():
    """Provides a mock Settings object returning a Pydantic-like config object."""
    settings = MagicMock(name="MockSettings")
    settings.model_dir = Path("/mock/models")

    # This mock simulates the PredictorConfig Pydantic model
    mock_predictor_config = MagicMock(spec=PredictorConfig)
    mock_predictor_config.sam_config_path = "sam/sam_config.json"
    mock_predictor_config.device = "cpu"
    mock_predictor_config.visualization = {"overlay_color": "red", "alpha": 0.4}

    settings.get_config.return_value = mock_predictor_config
    return settings


@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Mocks dependencies for all tests."""
    mocker.patch("culicidaelab.predictors.segmenter.build_sam2", return_value=MagicMock(name="mock_sam_model"))
    mocker.patch(
        "culicidaelab.predictors.segmenter.SAM2ImagePredictor",
        return_value=MagicMock(spec_set=["set_image", "predict"]),
    )
    mocker.patch("culicidaelab.predictors.segmenter.str_to_bgr", return_value=[0, 0, 255])

    mock_mwm_instance = MagicMock()
    mock_mwm_instance.ensure_weights.return_value = Path("/mock/models/segmenter_model.pth")
    mocker.patch("culicidaelab.core.base_predictor.ModelWeightsManager", return_value=mock_mwm_instance)


@pytest.fixture
def segmenter(mock_settings):
    """Provides a basic, un-loaded MosquitoSegmenter instance."""
    return MosquitoSegmenter(settings=mock_settings, load_model=False)


@pytest.fixture
def loaded_segmenter(segmenter):
    """Provides a MosquitoSegmenter with a mocked and 'loaded' model."""
    mock_internal_predictor = MagicMock(spec_set=["set_image", "predict"])
    segmenter._model = mock_internal_predictor
    segmenter._model_loaded = True
    segmenter.mocked_internal_predictor = mock_internal_predictor
    return segmenter


# --- Test Cases ---


def test_initialization(segmenter, mock_settings):
    """Test that the segmenter initializes correctly with a Pydantic config."""
    assert not segmenter.model_loaded
    assert isinstance(segmenter.config, MagicMock)
    assert segmenter.config.device == "cpu"
    mock_settings.get_config.assert_called_once_with("predictors.segmenter")


def test_load_model(segmenter, mocker):
    """Test the _load_model method using attribute access for config."""
    mock_build_sam2 = mocker.patch("culicidaelab.predictors.segmenter.build_sam2")
    segmenter._load_model()
    expected_config_path = str(segmenter.settings.model_dir / segmenter.config.sam_config_path)
    mock_build_sam2.assert_called_once_with(expected_config_path, segmenter.model_path, device=segmenter.config.device)


def test_predict_triggers_load_model(segmenter, mocker):
    """Test that predict() calls load_model() if the model is not loaded."""

    # Define a side effect that simulates the state changes of _load_model
    def load_model_side_effect():
        segmenter._model = MagicMock(spec_set=["set_image", "predict"])
        # Configure the mock to return a valid 3-tuple to avoid unpack errors
        segmenter._model.predict.return_value = (np.array([[[False]]]), [0.9], MagicMock())

    # Patch _load_model with the side effect
    mocked_load_model = mocker.patch.object(segmenter, "_load_model", side_effect=load_model_side_effect)

    # Act
    segmenter.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    # Assert
    mocked_load_model.assert_called_once()
    assert segmenter.model_loaded
    assert segmenter._model is not None
    segmenter._model.set_image.assert_called_once()


def test_predict_image_preprocessing(loaded_segmenter, mocker):
    """Test that input images are correctly converted to RGB."""
    mock_cv2 = mocker.patch("culicidaelab.predictors.segmenter.cv2")
    mock_cv2.cvtColor.side_effect = lambda img, flag: img

    # Configure the mock's return value to prevent the unpack error
    loaded_segmenter.mocked_internal_predictor.predict.return_value = (np.array([[[False]]]), [0.9], MagicMock())

    # Gray image
    gray_image = np.zeros((50, 50), dtype=np.uint8)
    loaded_segmenter.predict(gray_image)
    mock_cv2.cvtColor.assert_called_with(gray_image, mock_cv2.COLOR_GRAY2RGB)
    loaded_segmenter.mocked_internal_predictor.set_image.assert_called_with(gray_image)

    # RGBA image
    mock_cv2.cvtColor.reset_mock()
    rgba_image = np.zeros((50, 50, 4), dtype=np.uint8)
    loaded_segmenter.predict(rgba_image)
    mock_cv2.cvtColor.assert_called_with(rgba_image, mock_cv2.COLOR_RGBA2RGB)

    # RGB image (should not be converted)
    mock_cv2.cvtColor.reset_mock()
    rgb_image = np.zeros((50, 50, 3), dtype=np.uint8)
    loaded_segmenter.predict(rgb_image)
    mock_cv2.cvtColor.assert_not_called()


def test_predict_without_boxes_automatic_mode(loaded_segmenter):
    """Test prediction in automatic mode (no boxes provided)."""
    input_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_mask = np.zeros((100, 100), dtype=bool)
    mock_mask[10:20, 10:20] = True
    loaded_segmenter.mocked_internal_predictor.predict.return_value = ([mock_mask], [0.95], [MagicMock()])
    result_mask = loaded_segmenter.predict(input_image)
    loaded_segmenter.mocked_internal_predictor.set_image.assert_called_once_with(input_image)
    loaded_segmenter.mocked_internal_predictor.predict.assert_called_once_with(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=False,
    )
    np.testing.assert_array_equal(result_mask, mock_mask.astype(np.uint8))


def test_predict_with_detection_boxes(loaded_segmenter):
    """Test prediction when prompted with bounding boxes."""
    H, W = 100, 100
    input_image = np.zeros((H, W, 3), dtype=np.uint8)
    detection_boxes = [(20, 20, 20, 20, 0.9), (60, 60, 40, 40, 0.8)]
    mask1, mask2 = np.zeros((1, H, W), dtype=bool), np.zeros((1, H, W), dtype=bool)
    mask1[0, 15:25, 15:25], mask2[0, 50:70, 50:70] = True, True
    loaded_segmenter.mocked_internal_predictor.predict.side_effect = [
        (mask1, [0.91], [MagicMock()]),
        (mask2, [0.85], [MagicMock()]),
    ]
    result_mask = loaded_segmenter.predict(input_image, detection_boxes=detection_boxes)
    assert loaded_segmenter.mocked_internal_predictor.predict.call_count == 2
    calls = loaded_segmenter.mocked_internal_predictor.predict.call_args_list
    np.testing.assert_array_equal(calls[0].kwargs["box"], np.array([[10, 10, 30, 30]]))
    np.testing.assert_array_equal(calls[1].kwargs["box"], np.array([[40, 40, 80, 80]]))
    expected_combined_mask = np.logical_or(mask1[0], mask2[0])
    np.testing.assert_array_equal(result_mask, expected_combined_mask)


def test_visualize(segmenter, mocker):
    """Test the visualization method using attribute access for config."""
    mock_cv2 = mocker.patch("culicidaelab.predictors.segmenter.cv2")
    input_image = np.zeros((10, 10, 3), dtype=np.uint8)
    prediction_mask = np.ones((10, 10), dtype=np.uint8)
    alpha = segmenter.config.visualization["alpha"]
    overlay_color_str = segmenter.config.visualization["overlay_color"]
    mocker.patch("culicidaelab.predictors.segmenter.str_to_bgr", return_value=[0, 0, 255])
    mock_cv2.addWeighted.return_value = MagicMock()
    segmenter.visualize(input_image, prediction_mask)
    from culicidaelab.predictors.segmenter import str_to_bgr

    str_to_bgr.assert_called_once_with(overlay_color_str)
    mock_cv2.addWeighted.assert_called_once()
    assert mock_cv2.addWeighted.call_args[0][3] == alpha


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
    """Test the core metric calculation logic in _evaluate_from_prediction."""
    pred_mask = np.array(pred_data, dtype=np.uint8)
    gt_mask = np.array(gt_data, dtype=np.uint8)
    metrics = segmenter._evaluate_from_prediction(prediction=pred_mask, ground_truth=gt_mask)
    assert metrics["iou"] == pytest.approx(exp_iou)
    assert metrics["precision"] == pytest.approx(exp_prec)
    assert metrics["recall"] == pytest.approx(exp_rec)
    assert metrics["f1"] == pytest.approx(exp_f1)


def test_evaluate_public_method_integration(loaded_segmenter, mocker):
    """Test that the public evaluate method correctly calls predict and _evaluate_from_prediction."""
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)
    pred_mask = np.zeros((10, 10), dtype=np.uint8)
    mocker.patch.object(loaded_segmenter, "predict", return_value=pred_mask)
    mocker.patch.object(loaded_segmenter, "_evaluate_from_prediction", wraps=loaded_segmenter._evaluate_from_prediction)
    metrics = loaded_segmenter.evaluate(input_data=dummy_image, ground_truth=gt_mask)
    loaded_segmenter.predict.assert_called_once_with(dummy_image)
    loaded_segmenter._evaluate_from_prediction.assert_called_once_with(prediction=pred_mask, ground_truth=gt_mask)
    assert metrics["iou"] == 0.0
