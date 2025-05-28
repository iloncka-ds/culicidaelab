import pytest
import tempfile
from pathlib import Path
from culicidaelab.core.resource_manager import ResourceManager


@pytest.fixture
def resource_manager():
    """Fixture to create a ResourceManager instance with a temporary test directory."""
    return ResourceManager()


def test_initialization(resource_manager):
    """Test that ResourceManager initializes with correct directory structure."""
    assert resource_manager.app_name == "culicidaelab"
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
        from culicidaelab.core.resource_manager import ResourceManagerError

        with pytest.raises(ResourceManagerError):
            resource_manager.clean_temp_workspace(external_path)

        # Should work with force=True
        resource_manager.clean_temp_workspace(external_path, force=True)
