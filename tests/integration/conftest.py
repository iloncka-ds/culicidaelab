import pytest
import yaml
import shutil
from pathlib import Path

from culicidaelab.core.settings import Settings
from culicidaelab.core.resource_manager import ResourceManager


@pytest.fixture(scope="session")
def project_fixtures_dir() -> Path:
    """Returns the absolute path to the project's fixtures directory."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def create_dummy_model_files(project_fixtures_dir: Path):
    """Ensures dummy model files exist before any tests run."""
    project_fixtures_dir.mkdir(exist_ok=True)
    (project_fixtures_dir / "dummy_detector_model.pt").touch()
    (project_fixtures_dir / "dummy_segmenter_model.pt").touch()
    (project_fixtures_dir / "dummy_classifier_model.pkl").touch()


@pytest.fixture(scope="session")
def integration_test_dir(tmp_path_factory):
    base_dir = tmp_path_factory.mktemp("integration_tests")
    yield base_dir
    shutil.rmtree(base_dir)


@pytest.fixture(scope="session")
def resource_manager(integration_test_dir: Path) -> ResourceManager:
    return ResourceManager(app_name="culicidaelab_test", custom_base_dir=integration_test_dir)


@pytest.fixture
def user_config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "user_config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def settings_factory(monkeypatch, resource_manager):
    original_instance = Settings._instance

    def _create_settings(config_dir: str | Path | None = None) -> Settings:
        monkeypatch.setattr(Settings, "_instance", None)
        default_conf_path = Path(__file__).parent.parent.parent / "src" / "culicidaelab" / "conf"
        monkeypatch.setattr(
            "culicidaelab.core.config_manager.ConfigManager._get_default_config_path",
            lambda self: default_conf_path,
        )

        settings = Settings(config_dir=config_dir)
        settings._resource_manager = resource_manager
        return settings

    yield _create_settings
    monkeypatch.setattr(Settings, "_instance", original_instance)


def create_provider_config(config_dir: Path):
    """Creates a providers.yaml file with the correct structure."""
    provider_dict = {
        "huggingface": {
            "_target_": "culicidaelab.providers.huggingface_provider.HuggingFaceProvider",
            "dataset_url": "https://huggingface.co/api/datasets/{repo_id}",
            "api_key": None,
        },
    }
    with open(config_dir / "providers.yaml", "w") as f:
        yaml.dump(provider_dict, f)
