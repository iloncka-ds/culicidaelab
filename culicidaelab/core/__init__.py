"""Core components of the CulicidaeLab library.

This module provides the base classes, configuration management, and resource
handling functionalities that form the foundation of the library. It exports
key classes and functions for convenient access from other parts of the application.

Attributes:
    __all__ (list[str]): A list of the public objects of this module.
"""

# Base classes and protocols
from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.base_provider import BaseProvider
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend

# Configuration
from culicidaelab.core.config_manager import ConfigManager
from culicidaelab.core.config_models import (
    CulicidaeLabConfig,
    DatasetConfig,
    PredictorConfig,
    ProviderConfig,
    SpeciesModel,
)
from culicidaelab.core.species_config import SpeciesConfig
from culicidaelab.core.prediction_models import (
    BoundingBox,
    Detection,
    DetectionPrediction,
    SegmentationPrediction,
    Classification,
    ClassificationPrediction,
)

# Services and Managers
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.resource_manager import ResourceManager

# Settings Facade
from culicidaelab.core.settings import Settings, get_settings

# Utilities
from culicidaelab.core.utils import download_file

__all__ = [
    # Base classes and protocols
    "BasePredictor",
    "BaseProvider",
    "WeightsManagerProtocol",
    "BaseInferenceBackend",
    # Configuration
    "ConfigManager",
    "CulicidaeLabConfig",
    "PredictorConfig",
    "DatasetConfig",
    "ProviderConfig",
    "SpeciesModel",
    "SpeciesConfig",
    # Prediction Models
    "BoundingBox",
    "Detection",
    "DetectionPrediction",
    "SegmentationPrediction",
    "Classification",
    "ClassificationPrediction",
    # Services and Managers
    "ProviderService",
    "ResourceManager",
    # Settings Facade
    "Settings",
    "get_settings",
    # Utilities
    "download_file",
]
