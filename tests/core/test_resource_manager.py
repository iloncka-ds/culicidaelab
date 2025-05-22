import pytest
import tempfile
import time
import os
from pathlib import Path
from culicidaelab.core.resource_manager import ResourceManager


@pytest.fixture
def resource_manager():
    """Fixture to create a ResourceManager instance with a temporary test directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        rm = ResourceManager(app_name="test_culicidaelab")
        yield rm


def test_initialization(resource_manager):
    """Test that ResourceManager initializes with correct directory structure."""
    assert resource_manager.app_name == "test_culicidaelab"
    assert resource_manager.model_dir.exists()
    assert resource_manager.dataset_dir.exists()
    assert resource_manager.downloads_dir.exists()
    assert resource_manager.temp_dir.exists()


def test_get_model_path(resource_manager):
    """Test getting model path creates directory and returns correct path."""
    model_name = "test_model"
    model_path = resource_manager.get_model_path(model_name)
    assert model_path.exists()
    assert model_path.is_dir()
    assert model_path.name == model_name
    assert model_path.parent == resource_manager.model_dir


def test_get_dataset_path(resource_manager):
    """Test getting dataset path creates directory and returns correct path."""
    dataset_name = "test_dataset"
    dataset_path = resource_manager.get_dataset_path(dataset_name)
    assert dataset_path.exists()
    assert dataset_path.is_dir()
    assert dataset_path.name == dataset_name
    assert dataset_path.parent == resource_manager.dataset_dir


def test_create_temp_workspace(resource_manager):
    """Test creating temporary workspace."""
    workspace = resource_manager.create_temp_workspace(prefix="test_workspace")
    assert workspace.exists()
    assert workspace.is_dir()
    assert str(workspace).startswith(str(resource_manager.temp_dir))
    assert "test_workspace" in str(workspace)


def test_clean_temp_workspace(resource_manager):
    """Test cleaning temporary workspace."""
    # Create and verify workspace
    workspace = resource_manager.create_temp_workspace()
    assert workspace.exists()

    # Clean and verify removal
    resource_manager.clean_temp_workspace(workspace)
    assert not workspace.exists()


def test_clean_temp_workspace_safety(resource_manager):
    """Test that cleaning workspace outside temp directory raises error."""
    with tempfile.TemporaryDirectory() as external_dir:
        external_path = Path(external_dir)
        with pytest.raises(ValueError):
            resource_manager.clean_temp_workspace(external_path)

        # Should work with force=True
        resource_manager.clean_temp_workspace(external_path, force=True)


def test_clean_old_downloads(resource_manager):
    """Test cleaning old downloads."""
    # Create test files
    test_file = resource_manager.downloads_dir / "test_file.txt"
    test_file.touch()

    # Create test directory
    test_dir = resource_manager.downloads_dir / "test_dir"
    test_dir.mkdir()
    (test_dir / "inside_file.txt").touch()

    # Modify access time to make them old
    old_time = time.time() - (6 * 86400)  # 6 days old
    os.utime(test_file, (old_time, old_time))
    os.utime(test_dir, (old_time, old_time))

    # Clean files older than 5 days
    resource_manager.clean_old_downloads(days=5)

    # Verify cleanup
    assert not test_file.exists()
    assert not test_dir.exists()


def test_clean_old_downloads_keeps_new_files(resource_manager):
    """Test that cleaning old downloads preserves newer files."""
    # Create test file
    test_file = resource_manager.downloads_dir / "new_file.txt"
    test_file.touch()

    # Clean files older than 5 days
    resource_manager.clean_old_downloads(days=5)

    # Verify new file is preserved
    assert test_file.exists()
