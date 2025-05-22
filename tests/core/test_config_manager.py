"""Tests for ConfigManager class."""

import os
import pytest
from omegaconf import DictConfig

from culicidaelab.core.config_manager import ConfigManager


@pytest.fixture
def test_env_vars():
    """Fixture to set up test environment variables."""
    original_env = {}
    test_vars = {
        "HUGGINGFACE_API_KEY": "test_hf_key",
        "KAGGLE_API_KEY": "test_kaggle_key",
        "ROBOFLOW_API_KEY": "test_roboflow_key",
    }

    # Save original environment variables
    for key in test_vars:
        if key in os.environ:
            original_env[key] = os.environ[key]

    # Set test environment variables
    for key, value in test_vars.items():
        os.environ[key] = value

    yield test_vars

    # Restore original environment variables
    for key in test_vars:
        if key in original_env:
            os.environ[key] = original_env[key]
        else:
            del os.environ[key]


@pytest.fixture
def config_manager():
    """Fixture to create ConfigManager instance with test configuration."""
    # Use a simple relative path from the project root
    manager = ConfigManager("../../tests/conf")
    return manager


def test_singleton_pattern(config_manager):
    """Test that ConfigManager follows singleton pattern."""
    another_manager = ConfigManager()
    assert config_manager is another_manager


def test_load_config(config_manager):
    """Test loading configuration."""
    config = config_manager.load_config()
    assert isinstance(config, DictConfig)
    assert "providers" in config


def test_get_api_key(config_manager, test_env_vars):
    """Test getting API keys."""
    assert config_manager.get_api_key("huggingface") == test_env_vars["HUGGINGFACE_API_KEY"]
    assert config_manager.get_api_key("kaggle") == test_env_vars["KAGGLE_API_KEY"]
    assert config_manager.get_api_key("roboflow") == test_env_vars["ROBOFLOW_API_KEY"]


def test_get_provider_config(config_manager, test_env_vars):
    """Test getting provider configuration."""
    config = config_manager.load_config()
    provider_config = config_manager.get_provider_config("huggingface")

    assert isinstance(provider_config, dict)
    assert "api_key" in provider_config
    assert provider_config["api_key"] == test_env_vars["HUGGINGFACE_API_KEY"]


def test_instantiate(config_manager):
    """Test instantiating objects from config."""
    config = config_manager.load_config()
    # Add specific instantiation test based on your config structure
    assert config is not None


def test_config_overrides(config_manager):
    """Test configuration overrides."""
    config = config_manager.load_config(overrides=["+test_override=test_value"])
    assert config is not None
    assert "test_override" in config
    assert config.test_override == "test_value"
