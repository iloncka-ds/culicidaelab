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
    model_type = "detector"
    filename = "dummy_detector_model.pt"
    expected_relative_path = Path(model_type) / filename
    expected_absolute_path = resource_manager.model_dir / expected_relative_path

    detector_config_value = {
        "_target_": "culicidaelab.predictors.detector.MosquitoDetector",
        "model_path": str(expected_relative_path),
        "provider_name": "huggingface",
        "repository_id": "any/repo",
        "filename": filename,
        "model_config_path": None,
        "model_config_filename": None,
    }
    (user_config_dir / "predictors").mkdir()
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(detector_config_value, f)
    create_provider_config(user_config_dir)
    settings = settings_factory(config_dir=user_config_dir)

    mock_weights_manager_instance = MagicMock()

    def ensure_weights_side_effect(model_type_arg):
        expected_absolute_path.parent.mkdir(parents=True, exist_ok=True)
        expected_absolute_path.touch()
        return expected_absolute_path

    mock_weights_manager_instance.ensure_weights.side_effect = ensure_weights_side_effect

    monkeypatch.setattr(
        "culicidaelab.predictors.detector.ModelWeightsManager",
        MagicMock(return_value=mock_weights_manager_instance),
    )
    monkeypatch.setattr("culicidaelab.predictors.detector.ProviderService", MagicMock())

    detector = MosquitoDetector(settings=settings)

    with patch("culicidaelab.predictors.detector.YOLO") as mock_yolo_class:
        mock_yolo_instance = MagicMock()
        mock_yolo_class.return_value = mock_yolo_instance

        detector.load_model()

        mock_boxes = MagicMock(data=np.array([[10, 10, 50, 50, 0.9]]))
        mock_yolo_instance.predict.return_value = [MagicMock(boxes=mock_boxes)]
        detector.predict(np.zeros((128, 128, 3), dtype=np.uint8))

    assert detector.model_loaded
    mock_yolo_class.assert_called_once_with(str(expected_absolute_path), task="detect")


def test_full_segmenter_workflow(
    settings_factory,
    user_config_dir: Path,
    resource_manager: ResourceManager,
    monkeypatch,
):
    """Tests a full E2E scenario for the segmenter."""
    model_type = "segmenter"
    filename = "dummy_segmenter_model.pt"
    expected_relative_path = Path(model_type) / filename
    expected_absolute_path = resource_manager.model_dir / expected_relative_path

    segmenter_config_value = {
        "_target_": "culicidaelab.predictors.segmenter.MosquitoSegmenter",
        "model_path": str(expected_relative_path),
        "provider_name": "huggingface",
        "repository_id": "any/repo",
        "filename": filename,
        "model_config_path": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "model_config_filename": "sam2.1_hiera_t.yaml",
        "model_arch": "sam2.1_hiera_tiny",
        "device": "cpu",
    }
    (user_config_dir / "predictors").mkdir(exist_ok=True)
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(segmenter_config_value, f)

    create_provider_config(user_config_dir)
    settings = settings_factory(config_dir=user_config_dir)

    mock_weights_manager_instance = MagicMock()

    def ensure_weights_side_effect(model_type_arg):
        expected_absolute_path.parent.mkdir(parents=True, exist_ok=True)
        expected_absolute_path.touch()
        return expected_absolute_path

    mock_weights_manager_instance.ensure_weights.side_effect = ensure_weights_side_effect

    monkeypatch.setattr(
        "culicidaelab.predictors.segmenter.ModelWeightsManager",
        MagicMock(return_value=mock_weights_manager_instance),
    )
    monkeypatch.setattr("culicidaelab.predictors.segmenter.ProviderService", MagicMock())

    segmenter = MosquitoSegmenter(settings=settings)

    with patch("culicidaelab.predictors.segmenter.build_sam2") as mock_build_sam, patch(
        "culicidaelab.predictors.segmenter.SAM2ImagePredictor",
    ) as mock_predictor_class:
        mock_sam2_model_instance = MagicMock()
        mock_build_sam.return_value = mock_sam2_model_instance
        mock_predictor_instance = MagicMock()

        dummy_mask_3d = np.zeros((1, 100, 150), dtype=bool)
        mock_predictor_instance.predict.return_value = (dummy_mask_3d, MagicMock(), MagicMock())

        mock_predictor_class.return_value = mock_predictor_instance

        segmenter.load_model()
        result_mask = segmenter.predict(np.zeros((100, 150, 3), dtype=np.uint8))

    assert segmenter.model_loaded

    assert result_mask.shape == (100, 150)

    expected_mask_2d = np.zeros((100, 150), dtype=np.uint8)

    assert np.array_equal(result_mask, expected_mask_2d)

    mock_build_sam.assert_called_once()
    mock_predictor_class.assert_called_once_with(mock_sam2_model_instance)
    mock_predictor_instance.predict.assert_called_once()
