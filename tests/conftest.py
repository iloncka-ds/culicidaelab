"""
Pytest configuration file.
"""

import pytest
import os
import sys


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
