from culicidaelab.core.resource_manager import ResourceManager


def test_resource_manager_creates_directories(resource_manager: ResourceManager):
    """
    Tests that ResourceManager creates all necessary directories on initialization.
    """
    all_dirs = resource_manager.get_all_directories()
    for name, dir_path in all_dirs.items():
        assert dir_path.exists(), f"Directory '{name}' was not created at path {dir_path}"
        assert dir_path.is_dir()


def test_temp_workspace_creation_and_cleanup(resource_manager: ResourceManager):
    """
    Tests the full lifecycle of creating and automatically cleaning up a temporary workspace.
    """
    with resource_manager.temp_workspace("test_run") as workspace_path:
        assert workspace_path.exists()
        (workspace_path / "temp_file.txt").write_text("data")
        assert (workspace_path / "temp_file.txt").exists()

    assert not workspace_path.exists()
