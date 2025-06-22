import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch
from omegaconf import OmegaConf

# Import the new classes
from culicidaelab.predictors.detector import MosquitoDetector
from culicidaelab.core.settings import Settings


# Helper function to create a mock YOLO result
def create_mock_yolo_result(boxes_xyxy, confs):
    """Creates a mock YOLO result object that mimics ultralytics output."""
    mock_boxes = []
    for i, box_xyxy in enumerate(boxes_xyxy):
        mock_box = Mock()
        mock_box.xyxy = torch.tensor([box_xyxy])
        mock_box.conf = torch.tensor([confs[i]])
        mock_boxes.append(mock_box)

    mock_result = Mock()
    mock_result.boxes = mock_boxes
    return mock_result


@pytest.fixture
def mock_settings():
    """Fixture to create a mocked Settings instance for tests."""
    # This mock now replaces the old ConfigManager
    mock_settings_instance = Mock(spec=Settings)

    # Define the configuration that the detector will receive
    detector_config = {
        "confidence": 0.25,
        "params": {
            "iou_threshold": 0.45,
            "max_detections": 10,
        },
        "visualization": {
            "box_color": "#00FF00",
            "box_thickness": 2,
            "font_scale": 0.5,
            "text_color": "#FFFFFF",
            "text_thickness": 1,
        },
    }
    mock_detector_omega_conf = OmegaConf.create(detector_config)

    # Configure the mock to return this config when asked for 'predictors.detector'
    mock_settings_instance.get_config.return_value = mock_detector_omega_conf
    return mock_settings_instance


@pytest.fixture
def detector(mock_settings):
    """Create a MosquitoDetector instance with mocked dependencies."""
    # Patch ModelWeightsManager which is called in BasePredictor's __init__
    with patch("culicidaelab.core.base_predictor.ModelWeightsManager") as mock_weights_manager:
        # Ensure the manager returns a dummy path
        mock_weights_manager.return_value.ensure_weights.return_value = "dummy/path.pt"

        # Patch the YOLO model constructor to avoid loading the actual model
        with patch("culicidaelab.predictors.detector.YOLO") as mock_yolo_constructor:
            mock_yolo_instance = Mock()
            mock_yolo_constructor.return_value = mock_yolo_instance

            # Instantiate the detector using the new signature
            detector_instance = MosquitoDetector(
                settings=mock_settings,
                load_model=False,  # We will manually set the model
            )

            # Manually inject the mocked model and set its state
            detector_instance._model = mock_yolo_instance
            detector_instance._model_loaded = True

            return detector_instance


def test_detector_initialization(detector, mock_settings):
    """Test that the detector initializes correctly with the new structure."""
    assert detector.predictor_type == "detector"
    assert detector.config is not None
    # Check that settings.get_config was called correctly
    mock_settings.get_config.assert_called_once_with("predictors.detector")

    # Verify config values are set correctly
    assert detector.confidence_threshold == 0.25
    assert detector.config.params["iou_threshold"] == 0.45


def test_predict_single_image(detector):
    """Test detector prediction on a single image."""
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)

    # Mock the return value of the YOLO model call
    mock_yolo_result = create_mock_yolo_result(boxes_xyxy=[[100.0, 100.0, 200.0, 200.0]], confs=[0.9])
    detector._model.return_value = [mock_yolo_result]

    # Run prediction
    results = detector.predict(test_image)

    # Assertions
    assert len(results) == 1
    detection = results[0]
    # (center_x, center_y, width, height, confidence)
    assert detection[0] == pytest.approx(150.0)  # center_x
    assert detection[1] == pytest.approx(150.0)  # center_y
    assert detection[2] == pytest.approx(100.0)  # width
    assert detection[3] == pytest.approx(100.0)  # height
    assert detection[4] == pytest.approx(0.9)  # confidence

    # Check if the underlying model was called with the correct parameters
    detector._model.assert_called_once_with(
        source=test_image,
        conf=detector.confidence_threshold,
        iou=detector.config.params["iou_threshold"],
        max_det=detector.config.params["max_detections"],
        verbose=False,
    )


def test_predict_batch_efficiently(detector):
    """Test the overridden predict_batch method for efficient batch processing."""
    test_images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]

    # Mock the return value for a batch prediction
    # YOLO returns a list of results, one for each image
    mock_batch_results = [
        create_mock_yolo_result([[10, 10, 20, 20]], [0.9]),  # Image 1 result
        create_mock_yolo_result([], []),  # Image 2 result (no detections)
        create_mock_yolo_result([[30, 30, 40, 40], [50, 50, 60, 60]], [0.8, 0.85]),  # Image 3 result
    ]
    detector._model.return_value = mock_batch_results

    # Run batch prediction
    batch_predictions = detector.predict_batch(test_images, show_progress=False)

    # The model should be called ONCE with the entire batch
    detector._model.assert_called_once()
    assert detector._model.call_args[1]["source"] is test_images

    # Check the output structure
    assert isinstance(batch_predictions, list)
    assert len(batch_predictions) == 3

    # Check content of each prediction
    assert len(batch_predictions[0]) == 1
    assert batch_predictions[0][0][4] == pytest.approx(0.9)
    assert len(batch_predictions[1]) == 0
    assert len(batch_predictions[2]) == 2
    assert batch_predictions[2][1][4] == pytest.approx(0.85)


@pytest.mark.parametrize(
    "name, predictions, ground_truth, expected_metrics",
    [
        (
            "perfect match",
            [(50, 50, 20, 20, 0.9)],
            [(50, 50, 20, 20)],
            {"precision": 1.0, "recall": 1.0, "f1": 1.0, "ap": pytest.approx(1.0)},
        ),
        (
            "no predictions (false negative)",
            [],
            [(50, 50, 20, 20)],
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0},
        ),
        (
            "prediction, no GT (false positive)",
            [(50, 50, 20, 20, 0.9)],
            [],
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0},
        ),
        ("no prediction, no GT", [], [], {"precision": 1.0, "recall": 1.0, "f1": 1.0, "ap": 1.0}),
        (
            "one TP, one FP",
            [(50, 50, 20, 20, 0.9), (100, 100, 10, 10, 0.8)],  # Two predictions
            [(50, 50, 20, 20)],  # One GT
            {"precision": 0.5, "recall": 1.0, "f1": pytest.approx(0.666666666)},
        ),
    ],
    ids=["perfect_match", "false_negative", "false_positive", "empty_case", "mixed_tp_fp"],
)
def test_evaluate_from_prediction(detector, name, predictions, ground_truth, expected_metrics):
    """Test the core metric calculation logic in _evaluate_from_prediction."""
    metrics = detector._evaluate_from_prediction(predictions, ground_truth)

    for key, value in expected_metrics.items():
        assert metrics[key] == pytest.approx(value), f"Metric '{key}' failed for case '{name}'"


def test_evaluate_integration(detector):
    """
    Test the inherited `evaluate` method to ensure it correctly calls
    `predict` and `_evaluate_from_prediction`.
    """
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    ground_truth = [(10, 10, 5, 5)]
    mock_prediction = [(10, 10, 5, 5, 0.9)]

    # Patch the child-specific methods that `evaluate` will call
    with (
        patch.object(detector, "predict", return_value=mock_prediction) as mock_predict,
        patch.object(detector, "_evaluate_from_prediction") as mock_evaluate_logic,
    ):
        # Call the inherited evaluate method
        detector.evaluate(input_data=test_image, ground_truth=ground_truth)

        # Assert that the correct chain of methods was called
        mock_predict.assert_called_once_with(test_image)
        mock_evaluate_logic.assert_called_once_with(prediction=mock_prediction, ground_truth=ground_truth)


def test_evaluate_batch_integration(detector):
    """
    Test the inherited `evaluate_batch` method to ensure it correctly calls
    the efficient `predict_batch` and aggregates metrics.
    """
    test_images = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2
    gts = [[(10, 10, 5, 5)]] * 2
    mock_predictions = [[(10, 10, 5, 5, 0.9)]] * 2
    mock_metric = {"ap": 1.0}

    # Patch the child-specific methods
    with (
        patch.object(detector, "predict_batch", return_value=mock_predictions) as mock_predict_batch,
        patch.object(detector, "_evaluate_from_prediction", return_value=mock_metric) as mock_evaluate_logic,
    ):
        # Call the inherited batch evaluation method
        results = detector.evaluate_batch(
            input_data_batch=test_images,
            ground_truth_batch=gts,
            num_workers=1,  # Use 1 worker for deterministic testing
        )

        # Assert the correct methods were called
        mock_predict_batch.assert_called_once_with(test_images, show_progress=True)
        assert mock_evaluate_logic.call_count == 2

        # Check aggregation results
        assert results["ap"] == pytest.approx(1.0)
        assert results["ap_std"] == pytest.approx(0.0)
        assert results["ap_count"] == 2
