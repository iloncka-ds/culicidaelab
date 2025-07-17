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
        # Create a test file inside
        (workspace_path / "temp_file.txt").write_text("data")
        assert (workspace_path / "temp_file.txt").exists()

    # The directory should be deleted after exiting the context
    assert not workspace_path.exists()


def test_model_and_dataset_path_management(resource_manager: ResourceManager):
    """
    Tests that ResourceManager correctly creates paths for models and datasets.
    """
    # Get the model path
    model_path = resource_manager.get_model_path("my_test_model")
    assert model_path.exists()
    assert "my_test_model" in str(model_path)
    assert resource_manager.model_dir in model_path.parents

    # Create a file in the model directory
    (model_path / "weights.pt").touch()
    assert (model_path / "weights.pt").exists()

    # Get the dataset path
    dataset_path = resource_manager.get_dataset_path("my_test_dataset")
    assert dataset_path.exists()
    assert "my_test_dataset" in str(dataset_path)
    assert resource_manager.dataset_dir in dataset_path.parents
