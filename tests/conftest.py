"""
Pytest configuration file.
"""

import pytest
import os
import sys
from pathlib import Path
import tempfile
import shutil


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


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
