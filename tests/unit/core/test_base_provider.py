from pathlib import Path
from typing import Any
import pytest
from culicidaelab.core.base_provider import BaseProvider


def test_cannot_instantiate_abc():
    """Verify that BaseProvider, an ABC, cannot be instantiated directly."""
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class BaseProvider",
    ):
        BaseProvider()


def test_subclass_must_implement_methods():
    """Verify that a subclass must implement all abstract methods."""

    class IncompleteProvider(BaseProvider):
        def download_model_weights(
            self,
            model_type: str,
            *args: Any,
            **kwargs: Any,
        ) -> Path:
            return Path("/fake/weights")

        def get_provider_name(self) -> str:
            return "incomplete"

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class IncompleteProvider",
    ):
        IncompleteProvider()

    class CompleteProvider(BaseProvider):
        def download_dataset(
            self,
            dataset_name: str,
            save_dir: str | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> Path:
            return Path("/fake/dataset")

        def download_model_weights(
            self,
            model_type: str,
            *args: Any,
            **kwargs: Any,
        ) -> Path:
            return Path("/fake/weights")

        def get_provider_name(self) -> str:
            return "complete"

        def load_dataset(
            self,
            dataset_path: str | Path,
            split: str | None = None,
            **kwargs,
        ) -> any:
            return "fake_loaded_dataset"

    provider = CompleteProvider()
    assert provider is not None
    assert provider.get_provider_name() == "complete"
    assert provider.load_dataset(dataset_path="/fake/path") == "fake_loaded_dataset"
