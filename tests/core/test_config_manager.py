"""
Tests for ConfigManager class - functional testing approach.
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch
from pydantic import ValidationError
import tempfile
import shutil
from typing import Any

from culicidaelab.core.config_manager import ConfigManager, _deep_merge
from culicidaelab.core.config_models import CulicidaeLabConfig


# Fixtures
@pytest.fixture
def temp_config_dirs():
    """Create temporary directories for testing."""
    temp_dir = tempfile.mkdtemp()
    default_dir = Path(temp_dir) / "default"
    user_dir = Path(temp_dir) / "user"
    default_dir.mkdir(parents=True)
    user_dir.mkdir(parents=True)

    yield {"temp_dir": Path(temp_dir), "default_dir": default_dir, "user_dir": user_dir}

    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config_data():
    """Mock configuration data for testing."""
    return {
        "default": {
            "predictors": {
                "classifier": {
                    "target_": "culicidaelab.classifiers.DefaultClassifier",
                    "model_name": "default_model",
                    "threshold": 0.5,
                },
            },
            "datasets": {"batch_size": 32, "num_workers": 4},
        },
        "user": {
            "predictors": {"classifier": {"threshold": 0.7, "custom_param": "user_value"}},
            "datasets": {"batch_size": 64},
        },
    }


# Helper functions
def create_yaml_files(config_dir: Path, config_data: dict[str, Any]):
    """Helper to create YAML files in a directory."""
    for section, data in config_data.items():
        yaml_file = config_dir / f"{section}.yaml"
        with yaml_file.open("w") as f:
            yaml.dump(data, f)


# Tests for _deep_merge utility function
def test_deep_merge_simple_dict():
    """Test merging simple dictionaries."""
    source = {"a": 1, "b": 2}
    destination = {"b": 3, "c": 4}
    result = _deep_merge(source, destination)

    expected = {"a": 1, "b": 2, "c": 4}
    assert result == expected


def test_deep_merge_nested_dict():
    """Test merging nested dictionaries."""
    source = {"level1": {"level2": {"key1": "new_value", "key2": "source_value"}}}
    destination = {"level1": {"level2": {"key1": "old_value", "key3": "dest_value"}, "other_key": "preserved"}}
    result = _deep_merge(source, destination)

    expected = {
        "level1": {
            "level2": {"key1": "new_value", "key2": "source_value", "key3": "dest_value"},
            "other_key": "preserved",
        },
    }
    assert result == expected


def test_deep_merge_creates_new_keys():
    """Test that deep merge creates new nested keys."""
    source = {"new_section": {"new_key": "value"}}
    destination = {"existing": "value"}
    result = _deep_merge(source, destination)

    expected = {"new_section": {"new_key": "value"}, "existing": "value"}
    assert result == expected


def test_deep_merge_overwrites_non_dict_values():
    """Test that non-dict values are overwritten."""
    source = {"key": "new_value"}
    destination = {"key": {"nested": "old"}}
    result = _deep_merge(source, destination)

    expected = {"key": "new_value"}
    assert result == expected


# Tests for ConfigManager initialization
def test_config_manager_init_with_user_config_dir(temp_config_dirs, mock_config_data):
    """Test initialization with user config directory."""
    default_dir = temp_config_dirs["default_dir"]
    user_dir = temp_config_dirs["user_dir"]

    # Create config files
    create_yaml_files(default_dir, mock_config_data["default"])
    create_yaml_files(user_dir, mock_config_data["user"])

    with patch.object(ConfigManager, "_get_default_config_path", return_value=default_dir):
        with patch.object(ConfigManager, "_load", return_value=Mock(spec=CulicidaeLabConfig)):
            manager = ConfigManager(user_config_dir=str(user_dir))

            assert manager.user_config_dir == user_dir
            assert isinstance(manager.config, Mock)


def test_config_manager_init_without_user_config_dir(temp_config_dirs):
    """Test initialization without user config directory."""
    default_dir = temp_config_dirs["default_dir"]

    with patch.object(ConfigManager, "_get_default_config_path", return_value=default_dir):
        with patch.object(ConfigManager, "_load", return_value=Mock(spec=CulicidaeLabConfig)):
            manager = ConfigManager()

            assert manager.user_config_dir is None


def test_config_manager_pathlib_path_handling(temp_config_dirs):
    """Test that Path objects are handled correctly."""
    user_dir = temp_config_dirs["user_dir"]

    with patch.object(ConfigManager, "_get_default_config_path", return_value=Path("/mock")):
        with patch.object(ConfigManager, "_load", return_value=Mock(spec=CulicidaeLabConfig)):
            # Test with Path object
            manager = ConfigManager(user_config_dir=user_dir)
            assert isinstance(manager.user_config_dir, Path)
            assert manager.user_config_dir == user_dir

            # Test with string
            manager2 = ConfigManager(user_config_dir=str(user_dir))
            assert isinstance(manager2.user_config_dir, Path)
            assert manager2.user_config_dir == user_dir


# Tests for default config path resolution
def test_get_default_config_path_success():
    """Test successful retrieval of default config path."""
    mock_path = Path("/mock/path/conf")

    with patch("culicidaelab.core.config_manager.resources.files") as mock_resources:
        mock_resources.return_value.__truediv__.return_value = mock_path

        manager = ConfigManager.__new__(ConfigManager)  # Create without __init__
        result = manager._get_default_config_path()

        assert result == mock_path


def test_get_default_config_path_fallback():
    """Test fallback when resources.files fails."""
    with patch("culicidaelab.core.config_manager.resources.files", side_effect=ModuleNotFoundError):
        with patch("pathlib.Path.exists", return_value=True):
            manager = ConfigManager.__new__(ConfigManager)
            result = manager._get_default_config_path()

            # We can't easily test the exact path, so just ensure it's a Path object
            assert isinstance(result, Path)


def test_get_default_config_path_failure():
    """Test failure when both methods fail."""
    with patch("culicidaelab.core.config_manager.resources.files", side_effect=ModuleNotFoundError):
        with patch("pathlib.Path.exists", return_value=False):
            manager = ConfigManager.__new__(ConfigManager)

            with pytest.raises(FileNotFoundError, match="Could not find the default 'conf' directory"):
                manager._get_default_config_path()


# Tests for config loading from directories
def test_load_config_from_dir_success(temp_config_dirs, mock_config_data):
    """Test successful loading of config from directory."""
    config_dir = temp_config_dirs["default_dir"]
    create_yaml_files(config_dir, mock_config_data["default"])

    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(config_dir)

    assert "predictors" in result
    assert "datasets" in result
    assert result["predictors"]["classifier"]["model_name"] == "default_model"


def test_load_config_from_dir_nested_structure(temp_config_dirs):
    """Test loading config with nested directory structure."""
    config_dir = temp_config_dirs["default_dir"]

    # Create nested directory structure
    nested_dir = config_dir / "predictors"
    nested_dir.mkdir()

    classifier_config = {"target_": "test.Classifier", "param": "value"}
    detector_config = {"target_": "test.Detector", "param": "value"}

    with (nested_dir / "classifier.yaml").open("w") as f:
        yaml.dump(classifier_config, f)
    with (nested_dir / "detector.yaml").open("w") as f:
        yaml.dump(detector_config, f)

    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(config_dir)

    assert "predictors" in result
    assert "classifier" in result["predictors"]
    assert "detector" in result["predictors"]
    assert result["predictors"]["classifier"]["target_"] == "test.Classifier"


def test_load_config_from_dir_empty_yaml(temp_config_dirs):
    """Test handling of empty YAML files."""
    config_dir = temp_config_dirs["default_dir"]

    empty_file = config_dir / "empty.yaml"
    empty_file.write_text("")

    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(config_dir)

    # Empty files should be skipped
    assert "empty" not in result


def test_load_config_from_dir_invalid_yaml(temp_config_dirs, capsys):
    """Test handling of invalid YAML files."""
    config_dir = temp_config_dirs["default_dir"]

    invalid_file = config_dir / "invalid.yaml"
    invalid_file.write_text("invalid: yaml: content: [")

    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(config_dir)

    # Should print warning and continue
    captured = capsys.readouterr()
    assert "Warning: Could not load or parse" in captured.out
    assert "invalid" not in result


def test_load_config_from_dir_nonexistent():
    """Test loading from non-existent directory."""
    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(Path("/nonexistent/path"))

    assert result == {}


def test_load_config_from_dir_none():
    """Test loading with None directory."""
    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(None)

    assert result == {}


# Tests for config loading and validation
@patch.object(ConfigManager, "_load_config_from_dir")
def test_load_success(mock_load_config, temp_config_dirs):
    """Test successful configuration loading and validation."""
    # Mock the directory loading
    mock_load_config.side_effect = [
        {"predictors": {"classifier": {"target_": "test.Classifier"}}},  # default
        {"predictors": {"classifier": {"threshold": 0.8}}},  # user
    ]

    # Mock the CulicidaeLabConfig validation
    mock_validated_config = Mock(spec=CulicidaeLabConfig)

    with patch("culicidaelab.core.config_manager.CulicidaeLabConfig", return_value=mock_validated_config):
        manager = ConfigManager.__new__(ConfigManager)
        manager.default_config_path = temp_config_dirs["default_dir"]
        manager.user_config_dir = temp_config_dirs["user_dir"]

        result = manager._load()

        assert result == mock_validated_config
        # Verify both directories were loaded
        assert mock_load_config.call_count == 2


@patch.object(ConfigManager, "_load_config_from_dir")
def test_load_validation_error(mock_load_config, temp_config_dirs, capsys):
    """Test handling of validation errors during loading."""
    # Mock the directory loading
    mock_load_config.side_effect = [
        {"invalid": "config"},  # default
        {},  # user
    ]

    # Mock validation error
    validation_error = ValidationError.from_exception_data(
        "CulicidaeLabConfig",
        [{"type": "missing", "loc": ("required_field",), "msg": "Field required"}],
    )

    with patch("culicidaelab.core.config_manager.CulicidaeLabConfig", side_effect=validation_error):
        manager = ConfigManager.__new__(ConfigManager)
        manager.default_config_path = temp_config_dirs["default_dir"]
        manager.user_config_dir = temp_config_dirs["user_dir"]

        with pytest.raises(ValidationError):
            manager._load()

        # Verify error message was printed
        captured = capsys.readouterr()
        assert "FATAL: Configuration validation failed" in captured.out


# Tests for config retrieval and saving
def test_get_config():
    """Test getting the validated configuration."""
    mock_config = Mock(spec=CulicidaeLabConfig)

    manager = ConfigManager.__new__(ConfigManager)
    manager.config = mock_config

    result = manager.get_config()
    assert result == mock_config


def test_save_config(temp_config_dirs):
    """Test saving configuration to file."""
    save_path = temp_config_dirs["temp_dir"] / "saved" / "config.yaml"

    mock_config = Mock(spec=CulicidaeLabConfig)
    mock_config.model_dump.return_value = {"test": "config"}

    with patch("culicidaelab.core.config_manager.OmegaConf.save") as mock_save:
        manager = ConfigManager.__new__(ConfigManager)
        manager.config = mock_config

        manager.save_config(save_path)

        # Verify directory was created and OmegaConf.save was called
        assert save_path.parent.exists()
        mock_save.assert_called_once_with(config={"test": "config"}, f=save_path)


# Tests for object instantiation
def test_instantiate_from_config_success():
    """Test successful instantiation from config."""
    mock_config = Mock()
    mock_config.target_ = "builtins.dict"
    mock_config.model_dump.return_value = {"target_": "builtins.dict", "param1": "value1", "param2": "value2"}

    manager = ConfigManager.__new__(ConfigManager)
    result = manager.instantiate_from_config(mock_config, extra_param="extra")

    # Should create a dict with the merged parameters
    assert isinstance(result, dict)


def test_instantiate_from_config_no_target():
    """Test instantiation with missing target."""
    mock_config = Mock()
    mock_config.configure_mock(**{"target_": None})
    del mock_config.target_  # Remove the attribute

    manager = ConfigManager.__new__(ConfigManager)

    with pytest.raises(ValueError, match="Target key '_target_' not found"):
        manager.instantiate_from_config(mock_config)


def test_instantiate_from_config_invalid_target():
    """Test instantiation with invalid target."""
    mock_config = Mock()
    mock_config.target_ = "nonexistent.module.Class"
    mock_config.model_dump.return_value = {"target_": "nonexistent.module.Class"}

    manager = ConfigManager.__new__(ConfigManager)

    with pytest.raises(ImportError, match="Could not import and instantiate"):
        manager.instantiate_from_config(mock_config)


def test_instantiate_from_config_invalid_class():
    """Test instantiation with invalid class name."""
    mock_config = Mock()
    mock_config.target_ = "builtins.NonexistentClass"
    mock_config.model_dump.return_value = {"target_": "builtins.NonexistentClass"}

    manager = ConfigManager.__new__(ConfigManager)

    with pytest.raises(ImportError, match="Could not import and instantiate"):
        manager.instantiate_from_config(mock_config)


# Integration tests
def test_integration_full_workflow(temp_config_dirs, mock_config_data):
    """Test the full configuration workflow integration."""
    default_dir = temp_config_dirs["default_dir"]
    user_dir = temp_config_dirs["user_dir"]

    # Create realistic config files
    create_yaml_files(default_dir, mock_config_data["default"])
    create_yaml_files(user_dir, mock_config_data["user"])

    # Mock the validation to pass
    mock_config = Mock(spec=CulicidaeLabConfig)

    with patch.object(ConfigManager, "_get_default_config_path", return_value=default_dir):
        with patch("culicidaelab.core.config_manager.CulicidaeLabConfig", return_value=mock_config):
            manager = ConfigManager(user_config_dir=str(user_dir))

            # Verify the manager was initialized properly
            assert manager.user_config_dir == user_dir
            assert manager.default_config_path == default_dir
            assert manager.config == mock_config


# Edge case tests
def test_yaml_file_with_special_characters(tmp_path):
    """Test handling of YAML files with special characters in names."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Create file with special characters
    special_file = config_dir / "config-with-dashes_and_underscores.yaml"
    with special_file.open("w") as f:
        yaml.dump({"test": "value"}, f)

    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(config_dir)

    assert "config-with-dashes_and_underscores" in result


def test_deeply_nested_directory_structure(tmp_path):
    """Test deeply nested configuration directory structure."""
    config_dir = tmp_path / "config"
    deep_dir = config_dir / "level1" / "level2" / "level3"
    deep_dir.mkdir(parents=True)

    config_file = deep_dir / "deep_config.yaml"
    with config_file.open("w") as f:
        yaml.dump({"deep": "value"}, f)

    manager = ConfigManager.__new__(ConfigManager)
    result = manager._load_config_from_dir(config_dir)

    assert result["level1"]["level2"]["level3"]["deep_config"]["deep"] == "value"


def test_config_merge_with_complex_structures():
    """Test merging configurations with complex nested structures."""
    default_config = {
        "predictors": {
            "classifier": {
                "target_": "test.Classifier",
                "params": {"threshold": 0.5, "nested": {"deep_param": "default"}},
            },
        },
    }

    user_config = {"predictors": {"classifier": {"params": {"threshold": 0.8, "new_param": "user_value"}}}}

    result = _deep_merge(user_config, default_config)

    expected = {
        "predictors": {
            "classifier": {
                "target_": "test.Classifier",
                "params": {"threshold": 0.8, "new_param": "user_value", "nested": {"deep_param": "default"}},
            },
        },
    }

    assert result == expected
