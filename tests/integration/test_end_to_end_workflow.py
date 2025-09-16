import yaml
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from culicidaelab.core.resource_manager import ResourceManager
from culicidaelab.predictors.detector import MosquitoDetector
from culicidaelab.predictors.segmenter import MosquitoSegmenter
from .conftest import create_provider_config


def test_full_detector_workflow(
    settings_factory,
    user_config_dir: Path,
    resource_manager: ResourceManager,
    monkeypatch,
):
    """Tests a full E2E scenario for the detector."""
    # This test is already passing and needs no changes.
    model_type = "detector"
    filename = "dummy_detector_model.pt"
    expected_relative_path = Path(model_type) / filename
    expected_absolute_path = resource_manager.model_dir / expected_relative_path

    detector_config_value = {
        "target": "culicidaelab.predictors.detector.MosquitoDetector",
        "model_path": str(expected_relative_path),
        "provider_name": "huggingface",
        "repository_id": "any/repo",
        "filename": filename,
    }
    (user_config_dir / "predictors").mkdir()
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(detector_config_value, f)
    create_provider_config(user_config_dir)
    settings = settings_factory(config_dir=user_config_dir)

    mock_weights_manager_instance = MagicMock()

    def ensure_weights_side_effect(*args, **kwargs):
        expected_absolute_path.parent.mkdir(parents=True, exist_ok=True)
        expected_absolute_path.touch()
        return expected_absolute_path

    mock_weights_manager_instance.ensure_weights.side_effect = ensure_weights_side_effect
    monkeypatch.setattr(
        "culicidaelab.predictors.model_weights_manager.ModelWeightsManager",
        MagicMock(return_value=mock_weights_manager_instance),
    )

    detector = MosquitoDetector(settings=settings)

    # Note: The YOLO patch is correct because it's patching where YOLO is used.
    with patch("culicidaelab.predictors.backends.detector._yolo.YOLO") as mock_yolo_class:
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance
        detector.load_model()  # This call is now safe because YOLO is mocked.

    assert detector.model_loaded
    mock_yolo_class.assert_called_once_with(str(expected_absolute_path))


def test_full_segmenter_workflow(
    settings_factory,
    user_config_dir: Path,
    resource_manager: ResourceManager,
    monkeypatch,
):
    """Tests a full E2E scenario for the segmenter."""
    model_type = "segmenter"
    filename = "sam2.1_t.pt"
    expected_relative_path = Path(model_type) / filename
    expected_absolute_path = resource_manager.model_dir / expected_relative_path

    segmenter_config_value = {
        "target": "culicidaelab.predictors.segmenter.MosquitoSegmenter",
        "model_path": str(expected_relative_path),
        "provider_name": "huggingface",
        "repository_id": "any/repo",
        "filename": filename,
        "device": "cpu",
    }
    (user_config_dir / "predictors").mkdir(exist_ok=True)
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(segmenter_config_value, f)

    create_provider_config(user_config_dir)
    settings = settings_factory(config_dir=user_config_dir)

    mock_weights_manager_instance = MagicMock()

    def ensure_weights_side_effect(*args, **kwargs):
        expected_absolute_path.parent.mkdir(parents=True, exist_ok=True)
        # The empty file is still created, but our mock will prevent it from being used.
        expected_absolute_path.touch()
        return expected_absolute_path

    mock_weights_manager_instance.ensure_weights.side_effect = ensure_weights_side_effect
    monkeypatch.setattr(
        "culicidaelab.predictors.model_weights_manager.ModelWeightsManager",
        MagicMock(return_value=mock_weights_manager_instance),
    )

    segmenter = MosquitoSegmenter(settings=settings)

    with patch("culicidaelab.predictors.backends.segmenter._sam.SAM") as mock_sam_class:
        mock_sam_instance = MagicMock(name="mock_sam_instance")
        mock_result = MagicMock()
        # Correctly mock the nested structure
        mock_masks = MagicMock()
        mock_masks.data.cpu.return_value.numpy.return_value = np.zeros((1, 100, 150), dtype=bool)
        mock_result.masks = mock_masks
        mock_sam_instance.return_value = [mock_result]
        mock_sam_class.return_value = mock_sam_instance

        # This call is now safe because the real SAM constructor is mocked.
        segmenter.load_model()

        result_mask = segmenter.predict(
            np.zeros((100, 150, 3), dtype=np.uint8),
            detection_boxes=[(10, 10, 50, 50)],  # User-provided xyxy format
        )

    assert segmenter.model_loaded
    assert result_mask.mask.shape == (100, 150)
    # Assert that the mocked SAM class was instantiated.
    mock_sam_class.assert_called_once()
    # Assert that the mocked SAM instance was called by predict().
    mock_sam_instance.assert_called_once()
