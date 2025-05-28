from typing import Any
from datasets import load_dataset
from ..core.loader_protocol import DatasetLoader


class HuggingFaceDatasetLoader(DatasetLoader):
    """HuggingFace-specific dataset loader."""

    def load_dataset(self, path: str, split: str | None = None, **kwargs) -> Any:
        return load_dataset(path, split=split, **kwargs)
