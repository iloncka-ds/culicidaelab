import pytest

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


def test_get_dataset_path(resource_manager):
    """Test getting dataset path creates directory and returns correct path."""
    dataset_name = "test_dataset"
    dataset_path = resource_manager.get_dataset_path(dataset_name)
    assert dataset_path.exists()
    assert dataset_path.is_dir()
    assert dataset_path.name == dataset_name
    assert dataset_path.parent == resource_manager.dataset_dir


def test_temp_workspace_context(resource_manager):
    with resource_manager.temp_workspace(prefix="ctx") as ws:
        assert ws.exists()
    assert not ws.exists()


def test_clean_old_files(resource_manager):
    old_file = resource_manager.downloads_dir / "old.txt"
    old_file.write_text("x")
    old_time = time.time() - 10 * 86400
    os.utime(old_file, (old_time, old_time))
    stats = resource_manager.clean_old_files(days=5)
    assert stats["downloads_cleaned"] >= 1
    with pytest.raises(ValueError):
        resource_manager.clean_old_files(days=-1)


def test_get_disk_usage_and_directory_size(resource_manager):
    usage = resource_manager.get_disk_usage()
    assert "user_data" in usage
    assert resource_manager._get_directory_size(Path("not_a_dir"))["size_bytes"] == 0


def test_format_bytes(resource_manager):
    assert resource_manager._format_bytes(0) == "0 B"
    assert resource_manager._format_bytes(1024) == "1.0 KB"
    with pytest.raises(ValueError):
        resource_manager._format_bytes(None)


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
