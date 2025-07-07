import pytest
import tempfile

import os
import time
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
    workspace = resource_manager.create_temp_workspace()
    assert workspace.exists()

    resource_manager.clean_temp_workspace(workspace)
    assert not workspace.exists()


def test_clean_temp_workspace_safety(resource_manager):
    """Test that cleaning workspace outside temp directory raises error."""
    with tempfile.TemporaryDirectory() as external_dir:
        external_path = Path(external_dir)
        from culicidaelab.core.resource_manager import ResourceManagerError

        with pytest.raises(ResourceManagerError):
            resource_manager.clean_temp_workspace(external_path)

        resource_manager.clean_temp_workspace(external_path, force=True)


def test_get_cache_path_and_sanitize(resource_manager):
    cache_name = "test:cache*name?"
    cache_path = resource_manager.get_cache_path(cache_name)
    assert cache_path.exists()
    assert "_" in cache_path.name
    # Test ValueError for empty name
    with pytest.raises(ValueError):
        resource_manager.get_cache_path("")


def test_temp_workspace_context(resource_manager):
    with resource_manager.temp_workspace(prefix="ctx") as ws:
        assert ws.exists()
    assert not ws.exists()


def test_is_safe_to_delete(resource_manager):
    # Should be safe for temp_dir
    temp = resource_manager.create_temp_workspace()
    assert resource_manager._is_safe_to_delete(temp)
    # Should not be safe for root
    assert not resource_manager._is_safe_to_delete(Path("/"))


def test_clean_old_files(resource_manager):
    # Create an old file in downloads
    old_file = resource_manager.downloads_dir / "old.txt"
    old_file.write_text("x")
    old_time = time.time() - 10 * 86400
    os.utime(old_file, (old_time, old_time))
    stats = resource_manager.clean_old_files(days=5)
    assert stats["downloads_cleaned"] >= 1
    # Test ValueError for negative days
    with pytest.raises(ValueError):
        resource_manager.clean_old_files(days=-1)


def test_get_disk_usage_and_directory_size(resource_manager):
    usage = resource_manager.get_disk_usage()
    assert "user_data" in usage
    # Should handle non-existent path
    assert resource_manager._get_directory_size(Path("not_a_dir"))["size_bytes"] == 0


def test_format_bytes(resource_manager):
    assert resource_manager._format_bytes(0) == "0.0 B"
    assert resource_manager._format_bytes(1024) == "1.0 KB"
    with pytest.raises(ValueError):
        resource_manager._format_bytes(None)


def test_sanitize_name(resource_manager):
    assert resource_manager._sanitize_name("a:b/c*d?") == "a_b_c_d_"
    assert resource_manager._sanitize_name("   ...   ") == "unnamed"


def test_create_and_verify_checksum(resource_manager, tmp_path):
    file = tmp_path / "file.txt"
    file.write_text("abc")
    checksum = resource_manager.create_checksum(file, "md5")
    assert resource_manager.verify_checksum(file, checksum, "md5")
    assert not resource_manager.verify_checksum(file, "wrong", "md5")
    with pytest.raises(Exception):
        resource_manager.create_checksum(tmp_path / "notfound.txt")


def test_get_all_directories_and_repr(resource_manager):
    dirs = resource_manager.get_all_directories()
    assert "user_data_dir" in dirs
    assert "model_dir" in dirs
    assert "dataset_dir" in dirs
    assert isinstance(repr(resource_manager), str)
