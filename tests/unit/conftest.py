"""
Pytest configuration file.
"""

import pytest
import os
import sys
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock

from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.config_models import PredictorConfig


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(autouse=True)
def reset_global_settings_singleton(monkeypatch):
    """Ensure the settings singleton is reset before each test run."""
    import culicidaelab.core.settings

    monkeypatch.setattr(culicidaelab.core.settings, "_SETTINGS_INSTANCE", None)
    monkeypatch.setattr(culicidaelab.core.settings.Settings, "_initialized", False)


def pytest_collection_modifyitems(items):
    """Modify test collection to add markers."""
    for item in items:
        if "async" in item.nodeid:
            item.add_marker(pytest.mark.asyncio)


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test that may take longer to run",
    )


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def default_config_dir():
    """Return the path to the default config directory."""
    return Path(__file__).parent.parent / "culicidaelab" / "conf"


@pytest.fixture
def mock_weights_manager() -> Mock:
    """Fixture to provide a mock that adheres to WeightsManagerProtocol."""
    mock = Mock(spec=WeightsManagerProtocol)
    mock.ensure_weights.return_value = Path("mock/path/to/weights.pth")
    return mock


@pytest.fixture
def mock_provider_service() -> Mock:
    """Fixture to provide a mock ProviderService."""
    return Mock()


@pytest.fixture
def mock_settings(tmp_path: Path) -> Mock:
    """
    Provides a mock Settings object that returns a valid, structured
    PredictorConfig instance when its get_config method is called.
    """
    settings_mock = Mock()

    # Create a proper Pydantic model instance for the mock to return.
    mock_predictor_config_instance = PredictorConfig(
        target="some.dummy.class.path",
        model_path="mock/model.pkl",  # Added the required model_path
        backend="torch",
        repository_id="mock/repo",
        model_arch="test_arch",
        confidence=0.5,
        params={},
        weights={
            "torch": {"filename": "model.pkl"},
            "onnx": {"filename": "model.onnx"},
        },
    )

    # Mock the species config part
    species_config_mock = Mock()
    species_config_mock.species_map = {0: "species_A", 1: "species_B"}
    species_config_mock.class_to_full_name_map = {"species_A": "Species A", "species_B": "Species B"}
    inverse_map = {v: k for k, v in species_config_mock.species_map.items()}
    species_config_mock.get_index_by_species.side_effect = lambda name: inverse_map.get(name)

    # Configure the main settings mock
    settings_mock.species_config = species_config_mock
    settings_mock.get_config.return_value = mock_predictor_config_instance
    settings_mock.dataset_dir = tmp_path / "datasets"
    settings_mock.model_dir = tmp_path / "models"
    settings_mock.cache_dir = tmp_path / "cache"

    return settings_mock
