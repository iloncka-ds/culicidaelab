"""
Utility module for CulicidaeLab, providing various helper functions and classes.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any
from typing import cast

import cv2
import numpy as np
import torch
import yaml  # type: ignore
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset
from huggingface_hub import create_repo
from huggingface_hub import HfApi
from huggingface_hub import login
from huggingface_hub import Repository
from huggingface_hub import snapshot_download
from torch import device as torch_device


# Standard Library

# Third Party


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        Loaded image as numpy array
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, save_path: str | Path) -> None:
    """
    Save an image to file.

    Args:
        image: Image as numpy array
        save_path: Path to save image
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(save_path), image)


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file.

    Args:
        file_path: Path to YAML file

    Returns:
        Dictionary containing YAML data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path) as f:
        data = yaml.safe_load(f)

    return cast(dict[str, Any], data)


def save_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save data to a YAML file.

    Args:
        data: Data to save
        file_path: Path to save YAML file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(file_path: str | Path) -> dict[str, Any]:
    """
    Load a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary containing JSON data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path) as f:
        data = json.load(f)

    return cast(dict[str, Any], data)


def save_json(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to save JSON file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    keep_aspect_ratio: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize an image to target size.

    Args:
        image: Input image
        target_size: Target size (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio
        interpolation: Interpolation method

    Returns:
        Resized image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")

    if keep_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Create canvas of target size
        canvas = np.zeros((target_h, target_w) + image.shape[2:], dtype=image.dtype)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=interpolation)


def normalize_image(
    image: np.ndarray,
    mean: float | list[float] | None = None,
    std: float | list[float] | None = None,
) -> np.ndarray:
    """
    Normalize image values.

    Args:
        image: Input image
        mean: Mean values for normalization
        std: Standard deviation values for normalization

    Returns:
        Normalized image
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")

    # Convert to float32
    image = image.astype(np.float32)

    # Scale to [0, 1] if needed
    if image.max() > 1.0:
        image /= 255.0

    # Apply mean and std normalization
    if mean is not None and std is not None:
        mean = np.array(mean if isinstance(mean, list) else [mean])
        std = np.array(std if isinstance(std, list) else [std])
        image = (image - mean) / std

    return image


def create_directory(directory: str | Path) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = True,
) -> list[Path]:
    """
    List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: File pattern to match
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def ensure_dir(path: str | Path) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


class HuggingFaceManager:
    """Class to manage Hugging Face model and dataset interactions."""

    def __init__(self, token: str | None = None, cache_dir: str | None = None):
        """
        Initialize HuggingFace manager.

        Args:
            token (str, optional): HuggingFace API token. Only required for:
                                 - Accessing private repositories
                                 - Pushing models/datasets to hub
                                 - Saving to hub
                                 Public models and datasets can be accessed without a token.
            cache_dir (str, optional): Directory to cache models and datasets
        """
        self.token = token or os.environ.get("HUGGINGFACE_TOKEN")
        if self.token:
            login(token=self.token)

        self.cache_dir = cache_dir
        self.api = HfApi()

    def load_model(
        self,
        repo_id: str,
        model_class: Any,
        revision: str = "main",
        **model_kwargs,
    ) -> Any:
        """
        Load a model from Hugging Face Hub.
        No token required for public models.

        Args:
            repo_id (str): Repository ID (e.g., 'username/model-name')
            model_class: The model class to instantiate
            revision (str): Git revision to use
            **model_kwargs: Additional arguments to pass to model constructor

        Returns:
            Loaded model instance
        """
        try:
            # Try downloading with token first (if available)
            local_dir = snapshot_download(
                repo_id=repo_id,
                revision=revision,
                token=self.token,
                cache_dir=self.cache_dir,
            )
        except Exception as e:
            if "401 Client Error" in str(e) and self.token:
                # If token authentication failed, try without token for public repos
                local_dir = snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    token=None,
                    cache_dir=self.cache_dir,
                )
            else:
                raise e

        # Load model configuration if it exists
        config_path = Path(local_dir) / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            model_kwargs.update(config)

        # Load model weights
        weights_path = Path(local_dir) / "model.pth"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model = model_class(**model_kwargs)
            model.load_state_dict(state_dict)
            return model

        # If no weights file found, try loading directly through model class
        return model_class.from_pretrained(local_dir, **model_kwargs)

    def save_model(
        self,
        model: Any,
        repo_id: str,
        commit_message: str = "Update model",
        private: bool = False,
    ):
        """
        Save a model to Hugging Face Hub.
        Requires authentication token.

        Args:
            model: Model instance to save
            repo_id (str): Repository ID (e.g., 'username/model-name')
            commit_message (str): Commit message
            private (bool): Whether to create a private repository

        Raises:
            ValueError: If no authentication token is provided
        """
        if not self.token:
            raise ValueError("Authentication token required to save models to hub")

        # Create or clone repository
        repo_url = create_repo(repo_id, private=private, token=self.token, exist_ok=True)

        with Repository(local_dir="tmp_repo", clone_from=repo_url, token=self.token) as repo:
            # Save model configuration if available
            if hasattr(model, "config"):
                config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else vars(model.config)
                with open("config.yaml", "w") as f:
                    yaml.dump(config_dict, f)

            # Save model weights
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), "model.pth")
            else:
                model.save_pretrained(".")

            # Push to hub
            repo.push_to_hub(commit_message=commit_message)

    def load_dataset(
        self,
        repo_id: str,
        subset: str | None = None,
        split: str | None = None,
        **kwargs,
    ) -> Dataset | DatasetDict:
        """
        Load a dataset from Hugging Face Hub.
        No token required for public datasets.

        Args:
            repo_id (str): Repository ID (e.g., 'username/dataset-name')
            subset (str, optional): Dataset subset name
            split (str, optional): Which split of the dataset to load
            **kwargs: Additional arguments to pass to load_dataset

        Returns:
            Hugging Face dataset object
        """
        kwargs["cache_dir"] = self.cache_dir

        try:
            # Try loading with token first (if available)
            if self.token:
                kwargs["token"] = self.token
                return load_dataset(repo_id, subset, split=split, **kwargs)
        except Exception as e:
            if "401 Client Error" not in str(e):
                raise e

        # If token failed or wasn't provided, try without token for public datasets
        kwargs.pop("token", None)
        return load_dataset(repo_id, subset, split=split, **kwargs)

    def push_to_hub(
        self,
        dataset: Dataset | DatasetDict,
        repo_id: str,
        private: bool = False,
    ):
        """
        Push a dataset to Hugging Face Hub.
        Requires authentication token.

        Args:
            dataset: Dataset to push
            repo_id (str): Repository ID (e.g., 'username/dataset-name')
            private (bool): Whether to create a private repository

        Raises:
            ValueError: If no authentication token is provided
        """
        if not self.token:
            raise ValueError("Authentication token required to push datasets to hub")

        dataset.push_to_hub(repo_id, token=self.token, private=private)

    def list_models(
        self,
        owner: str | None = None,
        search: str | None = None,
        filter: str | None = None,
    ) -> list:
        """
        List available models on Hugging Face Hub.
        No token required.

        Args:
            owner (str, optional): Filter by model owner
            search (str, optional): Search query
            filter (str, optional): Filter string

        Returns:
            List of model info dictionaries
        """
        return self.api.list_models(author=owner, search=search, filter=filter)

    def list_datasets(self, owner: str | None = None, search: str | None = None) -> list:
        """
        List available datasets on Hugging Face Hub.
        No token required.

        Args:
            owner (str, optional): Filter by dataset owner
            search (str, optional): Search query

        Returns:
            List of dataset info dictionaries
        """
        return self.api.list_datasets(author=owner, search=search)


def get_device() -> torch_device:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        torch.device: Best available device
    """
    return torch_device("cuda" if torch.cuda.is_available() else "cpu")


def batch_process(items: list[Any], batch_size: int) -> list[list[Any]]:
    """
    Split a list of items into batches.

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
