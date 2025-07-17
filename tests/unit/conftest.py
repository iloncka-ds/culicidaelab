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


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


@pytest.fixture(autouse=True)
def reset_global_settings_singleton(monkeypatch):
    """Ensure the settings singleton is reset before each test run."""
    import culicidaelab.core.settings

    # Reset the singleton instance
    monkeypatch.setattr(culicidaelab.core.settings, "_SETTINGS_INSTANCE", None)
    # Reset the class-level initialization flag so that __init__ runs again
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
    """Fixture to provide a mock Settings object."""
    settings_mock = Mock()
    settings_mock.get_config.return_value = Mock()
    settings_mock.get_resource_dir.return_value = tmp_path
    settings_mock.cache_dir = tmp_path / "cache"
    settings_mock.user_data_dir = tmp_path / "user_data"
    settings_mock.model_dir = tmp_path / "models"
    return settings_mock
