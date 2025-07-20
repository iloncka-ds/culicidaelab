"""CulicidaeLab: A Python library for mosquito analysis.

This library provides a suite of tools for the detection, segmentation, and
classification of mosquitoes in images. It is designed to be a flexible
and extensible framework for researchers and developers working with
culicidae-related data.

This top-level package exposes the primary user-facing classes and functions
for easy access, including predictors, dataset managers, and configuration
utilities.

Main Components:
    - Predictors (`MosquitoDetector`, `MosquitoSegmenter`, `MosquitoClassifier`):
      Core models for performing analysis tasks.
    - `DatasetsManager`: A class for downloading and managing datasets.
    - `Settings`: A centralized configuration object for managing library behavior.
    - `SpeciesConfig`: Handles configuration related to mosquito species.
"""

from __future__ import annotations

# Core library components
from .core.base_predictor import BasePredictor
from .core.base_provider import BaseProvider
from .core.config_manager import ConfigManager
from .core.config_models import (
    CulicidaeLabConfig,
    DatasetConfig,
    PredictorConfig,
    ProviderConfig,
    SpeciesModel,
)
from .core.resource_manager import ResourceManager
from .core.settings import Settings, get_settings
from .core.species_config import SpeciesConfig
from .core.utils import download_file
from .core.weights_manager_protocol import WeightsManagerProtocol
from .core.provider_service import ProviderService

# Dataset management
from .datasets.datasets_manager import DatasetsManager

# Predictor models and managers
from .predictors.classifier import MosquitoClassifier
from .predictors.detector import MosquitoDetector
from .predictors.model_weights_manager import ModelWeightsManager
from .predictors.segmenter import MosquitoSegmenter

# Data providers
from .providers.huggingface_provider import HuggingFaceProvider

# Defines the public API for the package
__all__ = [
    "BasePredictor",
    "BaseProvider",
    "ConfigManager",
    "CulicidaeLabConfig",
    "PredictorConfig",
    "DatasetConfig",
    "ProviderConfig",
    "SpeciesModel",
    "SpeciesConfig",
    "DatasetsManager",
    "HuggingFaceProvider",
    "ModelWeightsManager",
    "MosquitoClassifier",
    "MosquitoDetector",
    "MosquitoSegmenter",
    "ProviderService",
    "ResourceManager",
    "Settings",
    "SpeciesConfig",
    "WeightsManagerProtocol",
    "download_file",
    "get_settings",
]


def __getattr__(name: str) -> str:
    """Lazily retrieves the library version using PEP 562."""
    if name != "__version__":
        msg = f"module {__name__} has no attribute {name}"
        raise AttributeError(msg)

    from importlib.metadata import version

    # Returns the version of the 'culicidaelab' package
    return version("culicidaelab")
