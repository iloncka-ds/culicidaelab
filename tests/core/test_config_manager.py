import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf

from culicidaelab.core.config_manager import ConfigManager, ConfigurableComponent


@pytest.fixture(autouse=True)
def reset_singleton():
    # Reset singleton before each test
    ConfigManager._instance = None


@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        # Write a config file with the expected structure
        with open(config_dir / "config.yaml", "w") as f:
            f.write(
                """
model:
  name: test_model
  params:
    batch_size: 32
data:
  path: /data
providers:
  kaggle:
    username: test_user
    competition: test_comp
""",
            )
        provider_dir = config_dir / "providers"
        provider_dir.mkdir(exist_ok=True)
        with open(provider_dir / "kaggle.yaml", "w") as f:
            f.write(
                """
username: test_user
competition: test_comp
""",
            )

        yield config_dir


@pytest.fixture
def mock_resource_manager():
    class DummyResourceManager:
        user_data_dir = Path("/tmp/data")
        user_cache_dir = Path("/tmp/cache")
        model_dir = Path("/tmp/models")
        dataset_dir = Path("/tmp/datasets")
        downloads_dir = Path("/tmp/downloads")

    with patch(
        "culicidaelab.core.config_manager.ResourceManager",
        DummyResourceManager,
    ):
        yield


@pytest.fixture
def config_manager(temp_config_dir, mock_resource_manager):
    # Patch _default_config_path to a non-existent directory to avoid loading library config
    with (
        patch.object(
            ConfigManager,
            "_default_config_path",
            str(temp_config_dir / "no_library"),
        ),
        patch.object(ConfigManager, "_get_project_root", return_value=temp_config_dir),
    ):
        manager = ConfigManager(config_path=str(temp_config_dir))
        manager.load_config()
        yield manager


def test_singleton_pattern(temp_config_dir, mock_resource_manager):
    with patch.object(
        ConfigManager,
        "_default_config_path",
        str(temp_config_dir / "no_library"),
    ):
        cm1 = ConfigManager(config_path=str(temp_config_dir))
        cm2 = ConfigManager(config_path=str(temp_config_dir))
        assert cm1 is cm2


def test_load_config(config_manager):
    config = config_manager.get_config()
    assert config.model.name == "test_model"
    assert config.data.path == "/data"


def test_get_config(config_manager):
    assert config_manager.get_config("model.name") == "test_model"
    assert config_manager.get_config("model.params.batch_size") == 32


def test_config_override(temp_config_dir, mock_resource_manager):
    with (
        patch.object(
            ConfigManager,
            "_default_config_path",
            str(temp_config_dir / "no_library"),
        ),
        patch.object(ConfigManager, "_get_project_root", return_value=temp_config_dir),
    ):
        manager = ConfigManager(config_path=str(temp_config_dir))
        overrides = {"model": {"name": "new_model"}}
        config = manager.load_config(overrides=overrides)
        assert config.model.name == "new_model"


def test_api_key_handling(config_manager):
    with patch.dict(os.environ, {"KAGGLE_API_KEY": "test_key"}):
        assert config_manager.get_api_key("kaggle") == "test_key"
    with pytest.raises(ValueError):
        config_manager.get_api_key("invalid_provider")


def test_provider_config(config_manager):
    with patch.dict(os.environ, {"KAGGLE_API_KEY": "test_key"}):
        config = config_manager.get_provider_config("kaggle")
        assert config["username"] == "test_user"
        assert config["competition"] == "test_comp"
        assert config["api_key"] == "test_key"


def test_merge_configs(config_manager):
    config1 = OmegaConf.create({"a": 1, "b": {"c": 2}})
    config2 = OmegaConf.create({"b": {"d": 3}, "e": 4})
    merged = config_manager._merge_configs(config1, config2)
    assert merged.a == 1
    assert merged.b.c == 2
    assert merged.b.d == 3
    assert merged.e == 4


def test_resource_dirs(config_manager):
    dirs = config_manager.get_resource_dirs()
    assert dirs["user_data_dir"] == Path("/tmp/data")
    assert dirs["cache_dir"] == Path("/tmp/cache")
    assert dirs["model_dir"] == Path("/tmp/models")
    assert dirs["dataset_dir"] == Path("/tmp/datasets")
    assert dirs["downloads_dir"] == Path("/tmp/downloads")
    assert dirs["archived_files_dir"] == Path("/tmp/downloads") / "archives"


def test_config_persistence(config_manager, temp_config_dir):
    config_manager.set_config_value("new.key", "test_value")
    save_path = temp_config_dir / "saved_config.yaml"
    config_manager.save_config(save_path)
    loaded_config = OmegaConf.load(save_path)
    assert loaded_config.new.key == "test_value"


def test_configuration_component(config_manager):
    component = ConfigurableComponent(config_manager)
    component.load_config()
    assert component.config.model.name == "test_model"
