import pytest
import yaml
import shutil
from pathlib import Path
from .conftest import create_provider_config
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager


def test_weights_manager_successful_download(
    settings_factory,
    user_config_dir: Path,
    monkeypatch,
    project_fixtures_dir: Path,
):
    """
    Tests that the ModelWeightsManager correctly orchestrates a download
    when the weights file does not exist.
    """
    create_provider_config(user_config_dir)

    model_type = "detector"
    backend_type = "yolo"
    filename = "dummy_detector_model.pt"

    detector_config_value = {
        "target": "culicidaelab.predictors.detector.MosquitoDetector",
        "provider_name": "huggingface",
        "repository_id": "test/dummy-model",
        "weights": {
            backend_type: {
                "filename": filename,
            },
        },
    }
    (user_config_dir / "predictors").mkdir(exist_ok=True)
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(detector_config_value, f)

    settings = settings_factory(user_config_dir)

    def mock_download(repo_id, filename, local_dir, **kwargs):
        # The weights manager will now construct the destination path itself.
        # We just need to copy the file to the expected location.
        dest_path = settings.model_dir / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(project_fixtures_dir / filename, dest_path)
        return str(dest_path)

    monkeypatch.setattr(
        "culicidaelab.providers.huggingface_provider.hf_hub_download",
        mock_download,
    )

    weights_manager = ModelWeightsManager(settings)

    final_path = weights_manager.ensure_weights(model_type, backend_type)

    expected_absolute_path = settings.model_dir / model_type / backend_type / filename

    assert final_path.exists()
    assert final_path == expected_absolute_path


def test_weights_manager_handles_download_failure(
    settings_factory,
    user_config_dir: Path,
    monkeypatch,
):
    """
    Tests that the ModelWeightsManager raises a RuntimeError if the provider fails.
    """
    create_provider_config(user_config_dir)

    model_type = "detector"
    backend_type = "yolo"
    filename = "model.pt"

    detector_config_value = {
        "target": "culicidaelab.predictors.detector.MosquitoDetector",
        "provider_name": "huggingface",
        "repository_id": "test/non-existent-model",
        "weights": {
            backend_type: {
                "filename": filename,
            },
        },
    }
    (user_config_dir / "predictors").mkdir(exist_ok=True)
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(detector_config_value, f)

    settings = settings_factory(user_config_dir)

    def mock_download_fails(*args, **kwargs):
        raise Exception("Simulated Hub Download Failure")

    monkeypatch.setattr(
        "culicidaelab.providers.huggingface_provider.hf_hub_download",
        mock_download_fails,
    )

    weights_manager = ModelWeightsManager(settings)

    with pytest.raises(
        RuntimeError,
        match=f"Failed to resolve weights for '{model_type}' with backend '{backend_type}'",
    ):
        weights_manager.ensure_weights(model_type, backend_type)
