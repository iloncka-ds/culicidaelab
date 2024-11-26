"""
Tests for the utility module.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import yaml  # type: ignore
from unittest.mock import patch, MagicMock
import cv2
from huggingface_hub import HfApi
import random

from culicidaelab.utils import (
    get_project_root,
    load_image,
    save_image,
    load_yaml,
    save_yaml,
    load_json,
    save_json,
    set_random_seed,
    resize_image,
    normalize_image,
    create_directory,
    list_files,
    ensure_dir,
    get_device,
    batch_process,
    HuggingFaceManager,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {"key": "value", "nested": {"key": "value"}}


def test_get_project_root():
    """Test get_project_root function."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()
    assert root.is_dir()


def test_load_image_with_valid_path(temp_dir):
    """Test loading a valid image."""
    # Create a test image
    img_path = temp_dir / "test.png"
    cv2.imwrite(str(img_path), np.zeros((100, 100, 3), dtype=np.uint8))

    loaded_img = load_image(img_path)
    assert isinstance(loaded_img, np.ndarray)
    assert loaded_img.shape == (100, 100, 3)


def test_load_image_with_invalid_path():
    """Test loading an invalid image."""
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent.png")


def test_save_image(temp_dir, sample_image):
    """Test saving an image."""
    save_path = temp_dir / "saved.png"
    save_image(sample_image, save_path)
    assert save_path.exists()

    # Verify the saved image
    loaded = cv2.imread(str(save_path))
    assert loaded is not None
    assert loaded.shape == sample_image.shape


def test_load_yaml_with_valid_file(temp_dir, sample_data):
    """Test loading a valid YAML file."""
    yaml_path = temp_dir / "test.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(sample_data, f)

    loaded_data = load_yaml(yaml_path)
    assert loaded_data == sample_data


def test_load_yaml_with_invalid_file():
    """Test loading an invalid YAML file."""
    with pytest.raises(FileNotFoundError):
        load_yaml("nonexistent.yaml")


def test_save_yaml(temp_dir, sample_data):
    """Test saving data to YAML."""
    yaml_path = temp_dir / "saved.yaml"
    save_yaml(sample_data, yaml_path)
    assert yaml_path.exists()

    # Verify the saved data
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)
    assert loaded == sample_data


def test_load_json_with_valid_file(temp_dir, sample_data):
    """Test loading a valid JSON file."""
    json_path = temp_dir / "test.json"
    with open(json_path, "w") as f:
        json.dump(sample_data, f)

    loaded_data = load_json(json_path)
    assert loaded_data == sample_data


def test_load_json_with_invalid_file():
    """Test loading an invalid JSON file."""
    with pytest.raises(FileNotFoundError):
        load_json("nonexistent.json")


def test_save_json(temp_dir, sample_data):
    """Test saving data to JSON."""
    json_path = temp_dir / "saved.json"
    save_json(sample_data, json_path)
    assert json_path.exists()

    # Verify the saved data
    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded == sample_data


def test_set_random_seed():
    """Test setting random seed."""
    set_random_seed(42)
    rand1 = random.random()
    torch_rand1 = torch.rand(1).item()
    np_rand1 = np.random.random()

    set_random_seed(42)
    rand2 = random.random()
    torch_rand2 = torch.rand(1).item()
    np_rand2 = np.random.random()

    assert rand1 == rand2
    assert torch_rand1 == torch_rand2
    assert np_rand1 == np_rand2


def test_resize_image_without_aspect_ratio(sample_image):
    """Test image resizing without maintaining aspect ratio."""
    target_size = (50, 75)
    resized = resize_image(sample_image, target_size, keep_aspect_ratio=False)
    assert resized.shape[:2] == (75, 50)  # Note: OpenCV uses (height, width) order


def test_resize_image_with_aspect_ratio(sample_image):
    """Test image resizing with aspect ratio preservation."""
    target_size = (50, 75)
    resized = resize_image(sample_image, target_size, keep_aspect_ratio=True)
    assert resized.shape[:2] == (75, 50)
    # Check for padding
    assert not np.any(resized[0, :])  # Top padding should be black
    assert not np.any(resized[-1, :])  # Bottom padding should be black


def test_normalize_image(sample_image):
    """Test image normalization."""
    # Test basic normalization
    normalized = normalize_image(sample_image)
    assert normalized.dtype == np.float32
    assert 0 <= normalized.max() <= 1.0

    # Test with mean and std
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    normalized = normalize_image(sample_image, mean=mean, std=std)
    assert normalized.dtype == np.float32


def test_create_directory(temp_dir):
    """Test directory creation."""
    new_dir = temp_dir / "new_dir"
    create_directory(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_list_files(temp_dir):
    """Test file listing."""
    # Create test files
    (temp_dir / "file1.txt").touch()
    (temp_dir / "file2.txt").touch()
    (temp_dir / "subdir").mkdir()
    (temp_dir / "subdir" / "file3.txt").touch()

    # Test non-recursive listing
    files = list_files(temp_dir, pattern="*.txt", recursive=False)
    assert len(files) == 2

    # Test recursive listing
    files = list_files(temp_dir, pattern="*.txt", recursive=True)
    assert len(files) == 3


def test_ensure_dir(temp_dir):
    """Test directory ensuring."""
    new_dir = temp_dir / "test_dir" / "nested_dir"
    ensure_dir(new_dir)
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_get_device():
    """Test device detection."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert str(device) in ["cpu", "cuda"]


def test_batch_process():
    """Test batch processing."""
    items = list(range(10))
    batches = batch_process(items, batch_size=3)
    assert len(batches) == 4  # 3, 3, 3, 1
    assert batches[0] == [0, 1, 2]
    assert batches[-1] == [9]


class TestHuggingFaceManager:
    """Tests for HuggingFaceManager class."""

    @pytest.fixture
    def manager(self):
        """Create a HuggingFaceManager instance."""
        return HuggingFaceManager()

    def test_init_without_token(self):
        """Test initialization without token."""
        manager = HuggingFaceManager()
        assert manager.token is None
        assert manager.cache_dir is None

    def test_init_with_token(self):
        """Test initialization with token."""
        with patch("culicidaelab.utils.login") as mock_login:
            manager = HuggingFaceManager(token="test_token")
            assert manager.token == "test_token"
            mock_login.assert_called_once_with(token="test_token")

    def test_load_model(self, manager):
        """Test model loading."""
        with patch("culicidaelab.utils.snapshot_download") as mock_download:
            mock_download.return_value = "mock_path"
            mock_model_class = MagicMock()

            manager.load_model("test/model", mock_model_class)
            mock_download.assert_called_once()
            mock_model_class.from_pretrained.assert_called_once()

    def test_list_models(self, manager):
        """Test model listing."""
        manager.list_models(owner="test_owner", search="test_query")
        assert isinstance(manager.api, HfApi)

    def test_list_datasets(self, manager):
        """Test dataset listing."""
        manager.list_datasets(owner="test_owner", search="test_query")
        assert isinstance(manager.api, HfApi)
