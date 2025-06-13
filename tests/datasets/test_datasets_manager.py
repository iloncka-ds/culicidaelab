import pytest
from unittest.mock import Mock
from culicidaelab.datasets.datasets_manager import DatasetsManager
from culicidaelab.datasets.huggingface import HuggingFaceDatasetLoader
from culicidaelab.core.config_manager import ConfigManager


@pytest.fixture
def mock_config_manager():
    return Mock(spec=ConfigManager)


@pytest.fixture
def mock_dataset_loader():
    loader = Mock(spec=HuggingFaceDatasetLoader)
    loader.load_dataset.return_value = {"data": "mock_dataset"}
    return loader


@pytest.fixture
def temp_config_file(tmp_path):
    config_content = """
    datasets:
      test_dataset:
        path: "test/dataset/path"
        description: "Test dataset"
    """
    config_file = tmp_path / "datasets_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def datasets_manager(mock_config_manager, mock_dataset_loader, temp_config_file):
    return DatasetsManager(
        config_manager=mock_config_manager,
        dataset_loader=mock_dataset_loader,
        config_path=temp_config_file,
    )


def test_load_config(datasets_manager):
    config = datasets_manager.load_config()
    assert "test_dataset" in config
    assert config["test_dataset"]["path"] == "test/dataset/path"


def test_add_dataset(datasets_manager):
    dataset_info = {"path": "new/dataset/path", "description": "New dataset"}
    datasets_manager.add_dataset("new_dataset", dataset_info)
    assert "new_dataset" in datasets_manager.datasets_config
    assert datasets_manager.datasets_config["new_dataset"] == dataset_info


def test_remove_dataset(datasets_manager):
    datasets_manager.remove_dataset("test_dataset")
    assert "test_dataset" not in datasets_manager.datasets_config


def test_update_dataset(datasets_manager):
    update_info = {"description": "Updated description"}
    datasets_manager.update_dataset("test_dataset", update_info)
    assert datasets_manager.datasets_config["test_dataset"]["description"] == "Updated description"


def test_load_dataset(datasets_manager, mock_dataset_loader):
    dataset = datasets_manager.load_dataset("test_dataset")
    mock_dataset_loader.load_dataset.assert_called_once_with(
        "test/dataset/path",
        split=None,
    )
    assert dataset == {"data": "mock_dataset"}


def test_get_loaded_dataset(datasets_manager):
    datasets_manager.load_dataset("test_dataset")
    dataset = datasets_manager.get_loaded_dataset("test_dataset")
    assert dataset == {"data": "mock_dataset"}


def test_load_dataset_missing_path(datasets_manager):
    datasets_manager.datasets_config["invalid_dataset"] = {}
    with pytest.raises(ValueError, match="Dataset path not specified"):
        datasets_manager.load_dataset("invalid_dataset")


def test_get_nonexistent_dataset(datasets_manager):
    with pytest.raises(KeyError, match="Dataset nonexistent not found"):
        datasets_manager.get_dataset_info("nonexistent")


def test_get_unloaded_dataset(datasets_manager):
    with pytest.raises(ValueError, match="Dataset test_dataset has not been loaded"):
        datasets_manager.get_loaded_dataset("test_dataset")


def test_list_datasets(datasets_manager):
    assert datasets_manager.list_datasets() == ["test_dataset"]
