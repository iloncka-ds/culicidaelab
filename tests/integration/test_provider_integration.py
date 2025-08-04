import pytest
import yaml
import shutil
from pathlib import Path
from .conftest import create_provider_config
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager
from culicidaelab.core.resource_manager import ResourceManager


def test_weights_manager_successful_download(
    settings_factory,
    user_config_dir: Path,
    monkeypatch,
    project_fixtures_dir: Path,
    resource_manager: ResourceManager,
):
    """
    Tests that the ModelWeightsManager correctly orchestrates a download
    when the weights file does not exist.
    """
    create_provider_config(user_config_dir)

    model_type = "detector"
    filename = "dummy_detector_model.pt"

    expected_relative_path = Path(model_type) / filename

    detector_config_value = {
        "_target_": "culicidaelab.predictors.detector.MosquitoDetector",
        "model_path": str(expected_relative_path),
        "provider_name": "huggingface",
        "repository_id": "test/dummy-model",
        "filename": filename,
        "model_config_path": None,
        "model_config_filename": None,
    }
    (user_config_dir / "predictors").mkdir(exist_ok=True)
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(detector_config_value, f)

    settings = settings_factory(user_config_dir)

    def mock_download(repo_id, filename, local_dir, **kwargs):
        dest_path = Path(local_dir) / filename
        shutil.copy(project_fixtures_dir / filename, dest_path)
        return str(dest_path)

    monkeypatch.setattr("culicidaelab.providers.huggingface_provider.hf_hub_download", mock_download)

    # ModelWeightsManager expects only the settings object; no separate ProviderService required.
    weights_manager = ModelWeightsManager(settings)

    final_path = weights_manager.ensure_weights(model_type)

    expected_absolute_path = resource_manager.model_dir / expected_relative_path

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
    detector_config_value = {
        "_target_": "...",
        "model_path": "detector/model.pt",
        "provider_name": "huggingface",
        "repository_id": "test/non-existent-model",
        "filename": "model.pt",
        "model_config_path": None,
        "model_config_filename": None,
    }
    (user_config_dir / "predictors").mkdir(exist_ok=True)
    with open(user_config_dir / "predictors" / f"{model_type}.yaml", "w") as f:
        yaml.dump(detector_config_value, f)

    settings = settings_factory(user_config_dir)

    def mock_download_fails(*args, **kwargs):
        raise Exception("Simulated Hub Download Failure")

    monkeypatch.setattr("culicidaelab.providers.huggingface_provider.hf_hub_download", mock_download_fails)

    # ModelWeightsManager expects only the settings object; no separate ProviderService required.
    weights_manager = ModelWeightsManager(settings)

    with pytest.raises(RuntimeError, match=f"Failed to download weights for '{model_type}'"):
        weights_manager.ensure_weights(model_type)
