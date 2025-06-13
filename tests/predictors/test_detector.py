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
            "device": "cpu",
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

    single_prediction_result = (100.0, 100.0, 100.0, 100.0, 0.8)

    with patch.object(detector, "_predict_batch", autospec=True) as mock_internal_predict_batch:
        mock_internal_predict_batch.side_effect = lambda sub_batch: [single_prediction_result for _ in sub_batch]

        results = detector.predict_batch(test_images, num_workers=1, batch_size=1)

    assert len(results) == 3, "Should get 3 prediction results for 3 images"
    assert mock_internal_predict_batch.call_count == 3

    for i, res_list_item in enumerate(results):
        assert res_list_item == single_prediction_result, f"Result for image {i} is incorrect"

    detector._model.reset_mock()

    with patch.object(detector, "predict", autospec=True) as mock_single_predict:
        mock_single_predict.return_value = single_prediction_result

        results = detector.predict_batch(test_images, num_workers=1, batch_size=1)

    assert len(results) == 3
    assert mock_single_predict.call_count == 3

    for res_item in results:
        assert res_item == single_prediction_result
