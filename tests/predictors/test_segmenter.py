import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock


try:
    import torch
except ImportError:
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.device = MagicMock(side_effect=lambda x: f"mock_device_{x.lower()}")
    sys.modules["torch"] = torch_mock

if "culicidaelab.core.config_manager" not in sys.modules:
    mock_config_manager_module = MagicMock()

    class DummyConfigManager:
        def get_config(self):
            return MockPyConfig()

    mock_config_manager_module.ConfigManager = DummyConfigManager
    sys.modules["culicidaelab.core.config_manager"] = mock_config_manager_module

if "culicidaelab.core.base_predictor" not in sys.modules:
    mock_base_predictor_module = MagicMock()

    class MockBasePredictorForInheritance:
        def __init__(self, model_path, config_manager):
            self.model_path = Path(model_path) if isinstance(model_path, str) else model_path
            self.config_manager = config_manager
            self.model = None
            self.model_loaded = False
            self.device = None
            self._config = self.config_manager.get_config()

        def load_model(self):
            if not self.model_loaded:
                self._load_model()
                self.model_loaded = True

        def _load_model(self):
            raise NotImplementedError("This should be overridden by MosquitoSegmenter")

    mock_base_predictor_module.BasePredictor = MockBasePredictorForInheritance
    sys.modules["culicidaelab.core.base_predictor"] = mock_base_predictor_module

if "sam2.build_sam" not in sys.modules:
    mock_build_sam_module = MagicMock()
    mock_build_sam_module.build_sam2 = MagicMock(name="global_mock_build_sam2")
    sys.modules["sam2.build_sam"] = mock_build_sam_module

if "sam2.sam2_image_predictor" not in sys.modules:
    mock_sam2_image_pred_module = MagicMock()
    mock_sam2_image_pred_module.SAM2ImagePredictor = MagicMock(
        name="GlobalMockSAM2ImagePredictor",
    )
    sys.modules["sam2.sam2_image_predictor"] = mock_sam2_image_pred_module

if "cv2" not in sys.modules:
    cv2_mock = MagicMock(name="global_cv2_mock")
    cv2_mock.COLOR_GRAY2RGB = "mock_COLOR_GRAY2RGB"
    cv2_mock.COLOR_RGBA2RGB = "mock_COLOR_RGBA2RGB"
    cv2_mock.COLOR_RGB2BGR = "mock_COLOR_RGB2BGR"
    sys.modules["cv2"] = cv2_mock


from culicidaelab.predictors.segmenter import MosquitoSegmenter


class MockPyConfig:
    """A mock configuration object mimicking the structure expected by MosquitoSegmenter."""

    def __init__(self):
        self.model = self._ModelConfig()
        self.visualization = self._VisualizationConfig()

    class _ModelConfig:
        def __init__(self):
            self.sam_config_path = "mock/sam_config.json"

    class _VisualizationConfig:
        def __init__(self):
            self.alpha = 0.5
            self.overlay_color = np.array([255, 0, 0], dtype=np.uint8)


MOCK_MODEL_PATH_STR = "mock_model.pth"


@pytest.fixture
def mock_config_obj():
    return MockPyConfig()


@pytest.fixture
def mock_config_manager_instance(mock_config_obj):
    ConfigManagerClass = sys.modules["culicidaelab.core.config_manager"].ConfigManager
    manager = MagicMock(spec=ConfigManagerClass)
    manager.get_config.return_value = mock_config_obj
    return manager


@pytest.fixture
def model_path():
    return Path(MOCK_MODEL_PATH_STR)


@pytest.fixture
def segmenter_instance(model_path, mock_config_manager_instance):
    """Basic MosquitoSegmenter instance, model not loaded."""
    return MosquitoSegmenter(
        model_path=model_path,
        config_manager=mock_config_manager_instance,
    )


@pytest.fixture
def loaded_segmenter_instance(segmenter_instance, mocker):
    """A MosquitoSegmenter instance with _load_model mocked and model marked as loaded."""
    mock_internal_sam_predictor = MagicMock(
        spec_set=["set_image", "generate", "predict"],
    )

    def _fake_load_model_side_effect():
        segmenter_instance.predictor = mock_internal_sam_predictor
        segmenter_instance.device = MagicMock(name="mock_device_from_loaded_fixture")

    mocker.patch.object(
        segmenter_instance,
        "_load_model",
        side_effect=_fake_load_model_side_effect,
    )

    segmenter_instance.load_model()

    segmenter_instance.mocked_internal_predictor = mock_internal_sam_predictor
    return segmenter_instance


def test_initialization(
    segmenter_instance,
    model_path,
    mock_config_manager_instance,
    mock_config_obj,
):
    assert segmenter_instance.model_path == model_path
    assert segmenter_instance.config_manager is mock_config_manager_instance
    assert segmenter_instance.model is None
    assert not segmenter_instance.model_loaded
    assert segmenter_instance.device is None
    assert segmenter_instance._config is mock_config_obj
    assert segmenter_instance.predictor is None


def test_load_model_on_cuda_if_available(
    mocker,
    model_path,
    mock_config_manager_instance,
    mock_config_obj,
):
    mock_torch_local = mocker.patch("culicidaelab.predictors.segmenter.torch")
    mock_torch_local.cuda.is_available.return_value = True
    mock_cuda_device_obj = MagicMock(name="cuda_device_obj_local")
    mock_torch_local.device.return_value = mock_cuda_device_obj

    mock_build_sam2_local = mocker.patch("culicidaelab.predictors.segmenter.build_sam2")
    mock_sam2_image_predictor_local = mocker.patch(
        "culicidaelab.predictors.segmenter.SAM2ImagePredictor",
    )

    mock_sam_model_instance = MagicMock(name="sam_model_instance_local")
    mock_build_sam2_local.return_value = mock_sam_model_instance
    mock_predictor_instance_local = MagicMock(name="sam2_predictor_instance_local")
    mock_sam2_image_predictor_local.return_value = mock_predictor_instance_local

    current_segmenter = MosquitoSegmenter(
        model_path=model_path,
        config_manager=mock_config_manager_instance,
    )
    segmenter_sam_config_path = mock_config_obj.model.sam_config_path

    current_segmenter._load_model()

    mock_torch_local.cuda.is_available.assert_called_once()
    mock_torch_local.device.assert_called_once_with("cuda")
    assert current_segmenter.device is mock_cuda_device_obj

    mock_build_sam2_local.assert_called_once_with(
        segmenter_sam_config_path,
        model_path,
        device=mock_cuda_device_obj,
    )
    mock_sam2_image_predictor_local.assert_called_once_with(mock_sam_model_instance)
    assert current_segmenter.predictor is mock_predictor_instance_local


def test_load_model_on_cpu_if_cuda_not_available(
    mocker,
    model_path,
    mock_config_manager_instance,
    mock_config_obj,
):
    mock_torch_local = mocker.patch("culicidaelab.predictors.segmenter.torch")
    mock_torch_local.cuda.is_available.return_value = False
    mock_cpu_device_obj = MagicMock(name="cpu_device_obj_local")
    mock_torch_local.device.return_value = mock_cpu_device_obj

    mock_build_sam2_local = mocker.patch("culicidaelab.predictors.segmenter.build_sam2")
    mock_sam2_image_predictor_local = mocker.patch(
        "culicidaelab.predictors.segmenter.SAM2ImagePredictor",
    )

    mock_sam_model_instance = MagicMock(name="sam_model_instance_local_cpu")
    mock_build_sam2_local.return_value = mock_sam_model_instance
    mock_predictor_instance_local = MagicMock(name="sam2_predictor_instance_local_cpu")
    mock_sam2_image_predictor_local.return_value = mock_predictor_instance_local

    current_segmenter = MosquitoSegmenter(
        model_path=model_path,
        config_manager=mock_config_manager_instance,
    )
    segmenter_sam_config_path = mock_config_obj.model.sam_config_path

    current_segmenter._load_model()

    mock_torch_local.cuda.is_available.assert_called_once()
    mock_torch_local.device.assert_called_once_with("cpu")
    assert current_segmenter.device is mock_cpu_device_obj
    mock_build_sam2_local.assert_called_once_with(
        segmenter_sam_config_path,
        model_path,
        device=mock_cpu_device_obj,
    )
    mock_sam2_image_predictor_local.assert_called_once_with(mock_sam_model_instance)
    assert current_segmenter.predictor is mock_predictor_instance_local


def test_predict_triggers_load_model_if_not_loaded(mocker, segmenter_instance):
    assert not segmenter_instance.model_loaded

    mock_internal_load_model = mocker.patch.object(segmenter_instance, "_load_model")

    mock_internal_sam_predictor = MagicMock(
        spec_set=["set_image", "generate", "predict"],
    )
    mock_internal_sam_predictor.generate.return_value = []

    def _load_model_side_effect():
        segmenter_instance.predictor = mock_internal_sam_predictor
        segmenter_instance.device = MagicMock(
            name="mock_device_from_trigger_side_effect",
        )

    mock_internal_load_model.side_effect = _load_model_side_effect

    dummy_input_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mocker.patch(
        "culicidaelab.predictors.segmenter.cv2.cvtColor",
        side_effect=lambda img, flag: img,
    )

    segmenter_instance.predict(dummy_input_image)

    mock_internal_load_model.assert_called_once()
    assert segmenter_instance.model_loaded
    assert segmenter_instance.predictor is mock_internal_sam_predictor
    segmenter_instance.predictor.set_image.assert_called_once_with(dummy_input_image)


def test_predict_image_preprocessing_conversions(mocker, loaded_segmenter_instance):
    mock_cv2_module = mocker.patch(
        "culicidaelab.predictors.segmenter.cv2",
    )

    rgb_shape = (10, 10, 3)
    converted_to_rgb_marker = np.full(rgb_shape, 123, dtype=np.uint8)
    mock_cv2_module.cvtColor.return_value = converted_to_rgb_marker

    loaded_segmenter_instance.mocked_internal_predictor.generate.return_value = []

    gray_image = np.zeros((10, 10), dtype=np.uint8)
    loaded_segmenter_instance.predict(gray_image)
    mock_cv2_module.cvtColor.assert_called_with(
        gray_image,
        mock_cv2_module.COLOR_GRAY2RGB,
    )
    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_with(
        converted_to_rgb_marker,
    )

    mock_cv2_module.cvtColor.reset_mock()
    loaded_segmenter_instance.mocked_internal_predictor.set_image.reset_mock()
    rgba_image = np.zeros((10, 10, 4), dtype=np.uint8)
    loaded_segmenter_instance.predict(rgba_image)
    mock_cv2_module.cvtColor.assert_called_with(
        rgba_image,
        mock_cv2_module.COLOR_RGBA2RGB,
    )
    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_with(
        converted_to_rgb_marker,
    )

    mock_cv2_module.cvtColor.reset_mock()
    loaded_segmenter_instance.mocked_internal_predictor.set_image.reset_mock()
    direct_rgb_image = np.zeros(rgb_shape, dtype=np.uint8)
    loaded_segmenter_instance.predict(direct_rgb_image)
    mock_cv2_module.cvtColor.assert_not_called()
    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_with(
        direct_rgb_image,
    )


def test_predict_with_detection_boxes(mocker, loaded_segmenter_instance):
    mocker.patch(
        "culicidaelab.predictors.segmenter.cv2.cvtColor",
        side_effect=lambda img, flag: img,
    )

    H, W = 100, 100
    input_image = np.zeros((H, W, 3), dtype=np.uint8)
    detection_boxes = [
        (10, 10, 20, 20, 0.9),
        (50, 50, 30, 30, 0.8),
    ]

    mask_data1 = np.zeros((1, H, W), dtype=bool)
    mask_data2 = np.zeros((1, H, W), dtype=bool)
    mask_data1[0, 15:25, 15:25] = True
    mask_data2[0, 55:75, 55:75] = True

    def predict_side_effect(*args, **kwargs):
        if np.array_equal(kwargs["box"], np.array([[10, 10, 30, 30]])):
            return mask_data1, "mock_score1", "mock_logits1"
        return mask_data2, "mock_score2", "mock_logits2"

    loaded_segmenter_instance.mocked_internal_predictor.predict.side_effect = predict_side_effect

    result_mask = loaded_segmenter_instance.predict(
        input_image,
        detection_boxes=detection_boxes,
    )

    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_once_with(
        input_image,
    )

    assert loaded_segmenter_instance.mocked_internal_predictor.predict.call_count == 2

    expected_combined_mask = np.logical_or(mask_data1, mask_data2)
    np.testing.assert_array_equal(result_mask, expected_combined_mask)
    assert result_mask.shape == (1, H, W)


def test_predict_with_empty_detection_boxes(loaded_segmenter_instance):
    H, W = 50, 60
    input_image = np.zeros((H, W, 3), dtype=np.uint8)
    loaded_segmenter_instance.mocked_internal_predictor.generate.return_value = []

    result_mask = loaded_segmenter_instance.predict(input_image, detection_boxes=[])

    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_once_with(
        input_image,
    )
    loaded_segmenter_instance.mocked_internal_predictor.predict.assert_not_called()

    expected_empty_mask = np.zeros((H, W), dtype=bool)
    np.testing.assert_array_equal(result_mask, expected_empty_mask)
    assert result_mask.shape == (H, W)


def test_predict_automatic_mask_generation_mode(mocker, loaded_segmenter_instance):
    mocker.patch(
        "culicidaelab.predictors.segmenter.cv2.cvtColor",
        side_effect=lambda img, flag: img,
    )

    H, W = 100, 100
    input_image = np.zeros((H, W, 3), dtype=np.uint8)

    auto_mask_data1 = np.zeros((H, W), dtype=bool)
    auto_mask_data1[10:20, 10:20] = True
    auto_mask_data2 = np.zeros((H, W), dtype=bool)
    auto_mask_data2[30:40, 30:40] = True

    loaded_segmenter_instance.mocked_internal_predictor.generate.return_value = [
        {"segmentation": auto_mask_data1, "other_meta": "foo"},
        {"segmentation": auto_mask_data2, "other_meta": "bar"},
    ]

    result_mask = loaded_segmenter_instance.predict(input_image, detection_boxes=None)

    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_once_with(
        input_image,
    )
    loaded_segmenter_instance.mocked_internal_predictor.generate.assert_called_once()

    expected_combined_auto_mask = np.logical_or(auto_mask_data1, auto_mask_data2)
    np.testing.assert_array_equal(result_mask, expected_combined_auto_mask)
    assert result_mask.shape == (H, W)


def test_predict_automatic_mode_with_no_masks_generated(loaded_segmenter_instance):
    loaded_segmenter_instance.mocked_internal_predictor.generate.return_value = []

    H, W = 50, 60
    input_image = np.zeros((H, W, 3), dtype=np.uint8)

    result_mask = loaded_segmenter_instance.predict(input_image)

    loaded_segmenter_instance.mocked_internal_predictor.set_image.assert_called_once_with(
        input_image,
    )
    loaded_segmenter_instance.mocked_internal_predictor.generate.assert_called_once()

    expected_empty_mask = np.zeros((H, W), dtype=bool)
    np.testing.assert_array_equal(result_mask, expected_empty_mask)
    assert result_mask.shape == (H, W)


def test_visualize_segmentation(mocker, segmenter_instance):
    mock_cv2_module = mocker.patch("culicidaelab.predictors.segmenter.cv2")

    H, W = 100, 100
    input_image_orig = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    prediction_mask = np.zeros((H, W), dtype=bool)
    prediction_mask[10:20, 10:20] = True

    num_true_pixels_in_mask = np.sum(prediction_mask)
    addweighted_result_pixels = np.full(
        (num_true_pixels_in_mask, 3),
        [10, 20, 30],
        dtype=np.uint8,
    )
    mock_cv2_module.addWeighted.return_value = addweighted_result_pixels

    viz_image = segmenter_instance.visualize(input_image_orig.copy(), prediction_mask)

    mock_cv2_module.addWeighted.assert_called_once()
    args = mock_cv2_module.addWeighted.call_args[0]
    np.testing.assert_array_equal(args[0], input_image_orig[prediction_mask])
    assert args[1] == segmenter_instance._config.visualization.alpha
    np.testing.assert_array_equal(
        args[2],
        segmenter_instance._config.visualization.overlay_color,
    )

    np.testing.assert_array_equal(viz_image[prediction_mask], addweighted_result_pixels)
    np.testing.assert_array_equal(
        viz_image[~prediction_mask],
        input_image_orig[~prediction_mask],
    )
    mock_cv2_module.imwrite.assert_not_called()

    mock_cv2_module.addWeighted.reset_mock()
    mock_cv2_module.cvtColor.reset_mock()
    mock_cv2_module.imwrite.reset_mock()

    bgr_converted_marker = np.full_like(input_image_orig, 77, dtype=np.uint8)
    mock_cv2_module.cvtColor.return_value = bgr_converted_marker

    temp_save_path = Path("test_visualization_output.png")
    viz_image_with_save = segmenter_instance.visualize(
        input_image_orig.copy(),
        prediction_mask,
        save_path=temp_save_path,
    )

    mock_cv2_module.addWeighted.assert_called_once()

    expected_image_for_cvtColor = input_image_orig.copy()
    expected_image_for_cvtColor[prediction_mask] = addweighted_result_pixels

    args_cvtColor, _ = mock_cv2_module.cvtColor.call_args
    np.testing.assert_array_equal(args_cvtColor[0], expected_image_for_cvtColor)
    assert args_cvtColor[1] == mock_cv2_module.COLOR_RGB2BGR

    mock_cv2_module.imwrite.assert_called_once_with(
        str(temp_save_path),
        bgr_converted_marker,
    )
    np.testing.assert_array_equal(viz_image_with_save, expected_image_for_cvtColor)


@pytest.mark.parametrize(
    "name, pred_mask_data, gt_mask_data, exp_iou, exp_prec, exp_rec, exp_f1",
    [
        (
            "Perfect match",
            [[True, True], [True, True]],
            [[True, True], [True, True]],
            1.0,
            1.0,
            1.0,
            1.0,
        ),
        (
            "No overlap (both non-empty)",
            [[True, False], [False, False]],
            [[False, False], [False, True]],
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            "Partial overlap",
            [[True, True], [False, False]],
            [[True, False], [True, False]],
            1 / 3,
            0.5,
            0.5,
            0.5,
        ),
        (
            "Empty prediction, non-empty ground truth",
            [[False, False], [False, False]],
            [[True, True], [True, True]],
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            "Non-empty prediction, empty ground truth",
            [[True, True], [True, True]],
            [[False, False], [False, False]],
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            "Both empty",
            [[False, False], [False, False]],
            [[False, False], [False, False]],
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    ],
)
def test_evaluate_segmentation_metrics(
    mocker,
    segmenter_instance,
    name,
    pred_mask_data,
    gt_mask_data,
    exp_iou,
    exp_prec,
    exp_rec,
    exp_f1,
):
    pred_mask = np.array(pred_mask_data, dtype=bool)
    gt_mask = np.array(gt_mask_data, dtype=bool)
    dummy_eval_input_image = np.zeros((2, 2, 3), dtype=np.uint8)

    mocker.patch.object(segmenter_instance, "predict", return_value=pred_mask)

    metrics = segmenter_instance.evaluate(dummy_eval_input_image, gt_mask)

    segmenter_instance.predict.assert_called_once_with(dummy_eval_input_image)

    assert metrics["iou"] == pytest.approx(exp_iou)
    assert metrics["precision"] == pytest.approx(exp_prec)
    assert metrics["recall"] == pytest.approx(exp_rec)
    assert metrics["f1"] == pytest.approx(exp_f1)
