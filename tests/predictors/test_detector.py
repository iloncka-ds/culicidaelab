import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch
from omegaconf import OmegaConf

from culicidaelab.predictors.detector import MosquitoDetector
from culicidaelab.core.config_manager import ConfigManager


@pytest.fixture
def config_manager():
    """Fixture to create ConfigManager instance for tests."""
    manager = ConfigManager(library_config_path="tests/conf")
    return manager


@pytest.fixture
def detector(config_manager):
    """Create a MosquitoDetector instance with mocked model and config."""
    mock_detector_component_config_dict = {
        "model": {
            "confidence_threshold": 0.25,
            "arch": "yolov11n",
            "iou_threshold": 0.45,
            "max_detections": 1,
            "device": "cpu",  # or None
        },
        "visualization": {
            "box_color": "#00FF00",
            "box_thickness": 2,
            "font_scale": 0.5,
            "text_color": [255, 255, 255],
            "text_thickness": 1,
        },
        "evaluation": {
            "iou_threshold": 0.5,
        },
    }
    mock_detector_component_omega_conf = OmegaConf.create(mock_detector_component_config_dict)
    config_manager.get_config = Mock(return_value=mock_detector_component_omega_conf)

    with patch("culicidaelab.predictors.detector.YOLO") as mock_yolo_class_constructor:
        mock_yolo_instance = Mock()
        mock_yolo_class_constructor.return_value = mock_yolo_instance

        detector_instance = MosquitoDetector(
            model_path="dummy/model/path.pt",
            config_manager=config_manager,
        )

        detector_instance._model = mock_yolo_instance
        detector_instance.model_loaded = True
        return detector_instance


def test_detector_initialization(detector):
    """Test that detector initializes with correct configuration."""
    assert detector.config is not None, "Detector config should be loaded."
    assert detector.config.model.arch == "yolov11n"
    assert detector.config.model.confidence_threshold == 0.25
    assert detector.config.model.iou_threshold == 0.45
    assert detector.config.model.max_detections == 1


def test_detector_predict_single_image(detector):
    """Test detector prediction on a single image."""
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)

    mock_individual_box = Mock()
    mock_individual_box.xyxy = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    mock_individual_box.conf = torch.tensor([0.9])

    mock_yolo_result_item = Mock()
    mock_yolo_result_item.boxes = [mock_individual_box]
    detector._model.return_value = [mock_yolo_result_item]

    results = detector.predict(test_image)

    assert len(results) == 1
    detection = results[0]
    assert detection[0] == pytest.approx(150.0)
    assert detection[1] == pytest.approx(150.0)
    assert detection[2] == pytest.approx(100.0)
    assert detection[3] == pytest.approx(100.0)
    assert detection[4] == pytest.approx(0.9)

    assert detector._model.called
    detector._model.assert_called_with(
        test_image,
        conf=detector.config.model.confidence_threshold,
        iou=detector.config.model.iou_threshold,
        max_det=detector.config.model.max_detections,
    )


def test_detector_predict_batch(detector):
    """Test detector prediction on a batch of images by mocking _predict_batch."""
    test_images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(3)]

    # Define what a single prediction result should look like
    # (center_x, center_y, w, h, conf)
    single_prediction_result = (100.0, 100.0, 100.0, 100.0, 0.8)

    # Mock the _predict_batch method of the detector instance directly.
    # This method is defined in BasePredictor and is what ThreadPoolExecutor calls.
    # It normally takes a list of images (a sub-batch) and returns a list of prediction results.
    with patch.object(detector, "_predict_batch", autospec=True) as mock_internal_predict_batch:
        # Configure the mock to return a list of predictions for each sub-batch it receives.
        # Since we're testing a batch of 3 images, and if batch_size=1 in predict_batch call,
        # _predict_batch would be called 3 times, each with one image.
        # Each call should return a list containing one prediction result.
        mock_internal_predict_batch.side_effect = lambda sub_batch: [single_prediction_result for _ in sub_batch]

        # Call the public predict_batch method
        # Using num_workers=1 and batch_size=1 to simplify the interaction with the mock.
        # The ThreadPoolExecutor will still be created, but its submit will call our mocked _predict_batch.
        results = detector.predict_batch(test_images, num_workers=1, batch_size=1)

    # Assertions
    assert len(results) == 3, "Should get 3 prediction results for 3 images"
    # Check that our mocked _predict_batch was called correctly.
    # With 3 images and batch_size=1, it should be called 3 times.
    assert mock_internal_predict_batch.call_count == 3

    # Check the content of the results
    for i, res_list_item in enumerate(results):
        # _predict_batch returns a list of predictions, and predict_batch extends results with these.
        # If _predict_batch returns [prediction_for_img1], then results becomes [prediction_for_img1, prediction_for_img2, ...]
        # So, `res_list_item` here is `single_prediction_result` directly.
        assert res_list_item == single_prediction_result, f"Result for image {i} is incorrect"

    # Additionally, we can check that the detector's actual `predict` method (and thus `_model`)
    # was NOT called, because we mocked `_predict_batch` which is higher up in that call chain for batch prediction.
    # However, `_predict_batch` in BasePredictor *does* call `self.predict`.
    # So, if we're testing that the batching logic in `BasePredictor.predict_batch` works,
    # and it correctly calls `_predict_batch`, then `_predict_batch` will call `self.predict`.
    # To test this properly, we'd need `_predict_batch` NOT to be mocked, but `self.predict` to be mocked.
    # Let's refine the test for better clarity of what's being tested.

    # OPTION 2: More robust test focusing on BasePredictor.predict_batch behavior
    # We mock `detector.predict` instead, which is called by `BasePredictor._predict_batch`.

    # Reset call counts if any from previous parts of the test function
    detector._model.reset_mock()  # Reset mock for the YOLO model instance

    # Mock the individual `predict` call that `_predict_batch` uses
    with patch.object(detector, "predict", autospec=True) as mock_single_predict:
        mock_single_predict.return_value = single_prediction_result  # `predict` returns a single result

        # Call the public predict_batch method
        results = detector.predict_batch(test_images, num_workers=1, batch_size=1)

    assert len(results) == 3
    assert mock_single_predict.call_count == 3  # `detector.predict` should be called for each image

    # Verify that the underlying YOLO model mock was also called if `detector.predict` wasn't fully mocked out
    # In our fixture, detector._model is a mock. If detector.predict wasn't mocked, this would be called.
    # Since we mocked detector.predict, detector._model should NOT have been called in this block.
    # detector._model.call_count should be 0 *for this specific call to predict_batch*.
    # This depends on how `detector.predict` is implemented and if it directly calls `self._model`.
    # In `MosquitoDetector.predict`, it does call `self._model`. So if `mock_single_predict` is a wrapper
    # around the real method, `self._model` would be called.
    # If `mock_single_predict` *replaces* the method, then `self._model` is not called by it.
    # `autospec=True` creates a mock with the same signature, but doesn't call the original.

    # Let's ensure that self._model (the yolo mock) is not called because `detector.predict` is fully mocked.
    # If the previous test `test_detector_predict_single_image` ran, `detector._model` has calls.
    # This assertion is tricky without isolating the test better or resetting the mock globally.
    # For now, focusing on mock_single_predict.call_count is the primary goal.

    for res_item in results:
        assert res_item == single_prediction_result
