from typing import Protocol, TypeVar

T = TypeVar("T", covariant=True)


class DatasetLoader(Protocol[T]):
    """Protocol for dataset loading strategies."""

    def load_dataset(self, path: str, split: str | None = None, **kwargs) -> T:
        """Load a dataset from the given path."""
        pass
