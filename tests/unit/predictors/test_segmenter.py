import sys
import os
import pytest
import numpy as np
from pathlib import Path

from unittest.mock import MagicMock
from importlib.util import find_spec

from culicidaelab.predictors.segmenter import MosquitoSegmenter

if find_spec("cv2") is not None:
    import cv2  # noqa
else:
    cv2_mock = MagicMock(name="global_cv2_mock")
    sys.modules["cv2"] = cv2_mock

if find_spec("sam2") is not None:
    import sam2  # noqa
else:
    sys.modules["sam2"] = MagicMock()
    sys.modules["sam2.build_sam"] = MagicMock()
    sys.modules["sam2.sam2_image_predictor"] = MagicMock()

try:
    from culicidaelab.core.config_models import PredictorConfig
except ImportError:
    PredictorConfig = type("PredictorConfig", (object,), {})
    if "culicidaelab.core.config_models" not in sys.modules:
        sys.modules["culicidaelab.core.config_models"] = MagicMock(
            PredictorConfig=PredictorConfig,
        )

sys.modules["culicidaelab.core.settings"] = MagicMock()
sys.modules["culicidaelab.core.utils"] = MagicMock()
sys.modules["culicidaelab.predictors.model_weights_manager"] = MagicMock()
sys.modules["culicidaelab.core.provider_service"] = MagicMock()


@pytest.fixture
def mock_settings():
    """Provides a mock Settings object returning a Pydantic-like config object."""
    settings = MagicMock(name="MockSettings")
    settings.model_dir = Path(r"c:/Users/lenova/CascadeProjects/culicidaelab")

    mock_predictor_config = MagicMock(spec=PredictorConfig)
    mock_predictor_config.model_config_path = "sam/sam_config.yaml"
    mock_predictor_config.device = "cpu"

    mock_viz_config = MagicMock()
    mock_viz_config.overlay_color = "red"
    mock_viz_config.alpha = 0.4
    mock_predictor_config.visualization = mock_viz_config

    settings.get_config.return_value = mock_predictor_config
    return settings


@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """Mocks dependencies for all tests."""
    mocker.patch(
        "culicidaelab.predictors.segmenter.build_sam2",
        return_value=MagicMock(name="mock_sam_model"),
    )
    mocker.patch(
        "culicidaelab.predictors.segmenter.SAM2ImagePredictor",
        return_value=MagicMock(spec_set=["set_image", "predict"]),
    )
    mocker.patch(
        "culicidaelab.predictors.segmenter.str_to_bgr",
        return_value=[0, 0, 255],
    )

    mock_mwm_instance = MagicMock()
    mock_mwm_instance.ensure_weights.return_value = Path(
        "/mock/models/segmenter_model.pth",
    )
    mocker.patch(
        "culicidaelab.predictors.segmenter.ModelWeightsManager",
        return_value=mock_mwm_instance,
    )


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


def test_initialization(segmenter, mock_settings):
    """Test that the segmenter initializes correctly with a Pydantic config."""
    assert not segmenter.model_loaded
    assert hasattr(segmenter, "config")
    assert segmenter.config.device == "cpu"
    mock_settings.get_config.assert_called_once_with("predictors.segmenter")


def test_load_model(segmenter, mocker):
    """Test the _load_model method using attribute access for config."""
    mock_build_sam2 = mocker.patch("culicidaelab.predictors.segmenter.build_sam2")
    mock_predictor = MagicMock()
    mocker.patch(
        "culicidaelab.predictors.segmenter.SAM2ImagePredictor",
        return_value=mock_predictor,
    )
    segmenter._load_model()
    expected_config_path = str(
        segmenter.settings.model_dir / segmenter.config.model_config_path,
    )
    mock_build_sam2.assert_called_once()
    args, kwargs = mock_build_sam2.call_args
    assert os.path.normcase(
        os.path.abspath(os.path.normpath(args[0])),
    ) == os.path.normcase(os.path.abspath(os.path.normpath(expected_config_path)))


def test_load_model_failure(segmenter, mocker):
    """Test that _load_model raises if build_sam2 fails."""
    mocker.patch(
        "culicidaelab.predictors.segmenter.build_sam2",
        side_effect=RuntimeError("fail"),
    )
    with pytest.raises(RuntimeError):
        segmenter._load_model()


def test_predict_raises_if_model_fails(segmenter, mocker):
    """Test predict raises if model cannot be loaded or is missing."""
    mocker.patch.object(segmenter, "load_model", side_effect=RuntimeError("fail"))
    with pytest.raises(RuntimeError):
        segmenter.predict(np.zeros((10, 10, 3), dtype=np.uint8))


def test_visualize_shape_mismatch(segmenter, mocker):
    """Test visualize raises if mask and image shapes do not match."""
    mocker.patch("culicidaelab.predictors.segmenter.cv2")
    input_image = np.zeros((10, 10, 3), dtype=np.uint8)
    prediction_mask = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(IndexError):
        segmenter.visualize(input_image, prediction_mask)


def test_evaluate_from_prediction_invalid_shapes(segmenter):
    """Test _evaluate_from_prediction with mismatched shapes raises."""
    pred_mask = np.ones((5, 5), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        segmenter._evaluate_from_prediction(prediction=pred_mask, ground_truth=gt_mask)


def test_evaluate_handles_exception(loaded_segmenter, mocker):
    """Test evaluate returns None or handles exception if predict fails."""
    mocker.patch.object(loaded_segmenter, "predict", side_effect=Exception("fail"))

    def mock_evaluate(input_data, ground_truth, **kwargs):
        try:
            prediction = loaded_segmenter.predict(input_data, **kwargs)
            return loaded_segmenter._evaluate_from_prediction(
                prediction=prediction,
                ground_truth=ground_truth,
            )
        except Exception:
            return None

    mocker.patch.object(loaded_segmenter, "evaluate", side_effect=mock_evaluate)
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)

    assert loaded_segmenter.evaluate(input_data=dummy_image, ground_truth=gt_mask) is None


def test_predict_triggers_load_model(segmenter, mocker):
    """Test that predict() calls load_model() if the model is not loaded."""

    def load_model_side_effect():
        segmenter._model = MagicMock(spec_set=["set_image", "predict"])
        segmenter._model.predict.return_value = (
            np.array([[[False]]]),
            [0.9],
            MagicMock(),
        )
        segmenter._model_loaded = True

    mocked_load_model = mocker.patch.object(
        segmenter,
        "load_model",
        side_effect=load_model_side_effect,
    )

    segmenter.predict(np.zeros((10, 10, 3), dtype=np.uint8))

    mocked_load_model.assert_called_once()
    assert segmenter.model_loaded
    assert segmenter._model is not None
    segmenter._model.set_image.assert_called_once()


def test_predict_image_preprocessing(loaded_segmenter, mocker):
    """Test that input images are correctly converted to RGB."""
    mock_cv2 = mocker.patch("culicidaelab.predictors.segmenter.cv2")
    mock_cv2.cvtColor.side_effect = lambda img, flag: img

    loaded_segmenter.mocked_internal_predictor.predict.return_value = (
        np.array([[[False]]]),
        [0.9],
        MagicMock(),
    )

    gray_image = np.zeros((50, 50), dtype=np.uint8)
    loaded_segmenter.predict(gray_image)
    mock_cv2.cvtColor.assert_called_with(gray_image, mock_cv2.COLOR_GRAY2RGB)
    loaded_segmenter.mocked_internal_predictor.set_image.assert_called_with(gray_image)

    mock_cv2.cvtColor.reset_mock()
    rgba_image = np.zeros((50, 50, 4), dtype=np.uint8)
    loaded_segmenter.predict(rgba_image)
    mock_cv2.cvtColor.assert_called_with(rgba_image, mock_cv2.COLOR_RGBA2RGB)

    mock_cv2.cvtColor.reset_mock()
    rgb_image = np.zeros((50, 50, 3), dtype=np.uint8)
    loaded_segmenter.predict(rgb_image)
    mock_cv2.cvtColor.assert_not_called()


def test_predict_without_boxes(loaded_segmenter):
    """Test prediction in automatic mode (no boxes provided)."""
    input_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_mask = np.zeros((100, 100), dtype=bool)
    mock_mask[10:20, 10:20] = True
    loaded_segmenter.mocked_internal_predictor.predict.return_value = (
        [mock_mask],
        [0.95],
        [MagicMock()],
    )
    result_mask = loaded_segmenter.predict(input_image)
    loaded_segmenter.mocked_internal_predictor.set_image.assert_called_once_with(
        input_image,
    )
    loaded_segmenter.mocked_internal_predictor.predict.assert_called_once_with(
        point_coords=None,
        point_labels=None,
        box=None,
        multimask_output=False,
    )
    np.testing.assert_array_equal(result_mask, mock_mask.astype(np.uint8))


def test_predict_with_boxes(loaded_segmenter):
    """Test prediction when prompted with bounding boxes."""
    H, W = 100, 100
    input_image = np.zeros((H, W, 3), dtype=np.uint8)
    detection_boxes = [(20, 20, 20, 20, 0.9), (60, 60, 40, 40, 0.8)]
    mask1 = np.zeros((1, H, W), dtype=bool)
    mask1[0, 15:25, 15:25] = True
    mask2 = np.zeros((1, H, W), dtype=bool)
    mask2[0, 50:70, 50:70] = True
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
    alpha = segmenter.config.visualization.alpha
    overlay_color_str = segmenter.config.visualization.overlay_color
    str_to_bgr_mock = mocker.patch(
        "culicidaelab.predictors.segmenter.str_to_bgr",
        return_value=[0, 0, 255],
    )
    mock_cv2.addWeighted.return_value = MagicMock()
    segmenter.visualize(input_image, prediction_mask)

    str_to_bgr_mock.assert_called_once_with(overlay_color_str)
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
def test_evaluate_from_prediction(
    segmenter,
    name,
    pred_data,
    gt_data,
    exp_iou,
    exp_prec,
    exp_rec,
    exp_f1,
):
    """Test the core metric calculation logic in _evaluate_from_prediction."""
    pred_mask = np.array(pred_data, dtype=np.uint8)
    gt_mask = np.array(gt_data, dtype=np.uint8)
    metrics = segmenter._evaluate_from_prediction(
        prediction=pred_mask,
        ground_truth=gt_mask,
    )
    assert metrics["iou"] == pytest.approx(exp_iou)
    assert metrics["precision"] == pytest.approx(exp_prec)
    assert metrics["recall"] == pytest.approx(exp_rec)
    assert metrics["f1"] == pytest.approx(exp_f1)


def test_evaluate_integration(loaded_segmenter, mocker):
    """Test that the public evaluate method correctly calls predict and _evaluate_from_prediction."""
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    gt_mask = np.ones((10, 10), dtype=np.uint8)
    pred_mask = np.zeros((10, 10), dtype=np.uint8)

    mocker.patch.object(loaded_segmenter, "predict", return_value=pred_mask)
    mocker.patch.object(
        loaded_segmenter,
        "_evaluate_from_prediction",
        wraps=loaded_segmenter._evaluate_from_prediction,
    )

    def mock_evaluate(input_data, ground_truth, **kwargs):
        prediction = loaded_segmenter.predict(input_data, **kwargs)
        return loaded_segmenter._evaluate_from_prediction(
            prediction=prediction,
            ground_truth=ground_truth,
        )

    mocker.patch.object(loaded_segmenter, "evaluate", side_effect=mock_evaluate)

    metrics = loaded_segmenter.evaluate(input_data=dummy_image, ground_truth=gt_mask)

    loaded_segmenter.predict.assert_called_once_with(dummy_image)
    loaded_segmenter._evaluate_from_prediction.assert_called_once_with(
        prediction=pred_mask,
        ground_truth=gt_mask,
    )
    assert metrics["iou"] == 0.0
    loaded_segmenter._evaluate_from_prediction.assert_called_once_with(
        prediction=pred_mask,
        ground_truth=gt_mask,
    )
    assert metrics["iou"] == 0.0
