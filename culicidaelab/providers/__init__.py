"""Data provider implementations for accessing datasets and models.

This package contains classes that implement the `BaseProvider` interface
to interact with various data sources like Hugging Face, Kaggle, etc.
Each provider module offers specific logic for downloading datasets
and model weights.

Available Classes:
    - HuggingFaceProvider: A provider for interacting with the Hugging Face Hub.
"""

from culicidaelab.providers.huggingface_provider import HuggingFaceProvider

__all__ = [
    "HuggingFaceProvider",
]
