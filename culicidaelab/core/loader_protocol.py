from typing import Protocol, TypeVar

T = TypeVar("T")


class DatasetLoader(Protocol[T]):
    """Protocol for dataset loading strategies."""

    def load_dataset(self, path: str, split: str | None = None, **kwargs) -> T:
        """Load a dataset from the given path."""
        pass
