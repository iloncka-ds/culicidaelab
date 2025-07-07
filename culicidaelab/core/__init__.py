"""
Core components of the CulicidaeLab library.

This module provides the base classes, configuration management,
and resource handling functionalities that form the foundation of the library.
"""

from .base_predictor import BasePredictor
from .base_provider import BaseProvider
from .config_manager import ConfigManager
from .config_models import (
    CulicidaeLabConfig,
    PredictorConfig,
    DatasetConfig,
    ProviderConfig,
    SpeciesModel,
)
from .provider_service import ProviderService
from .resource_manager import ResourceManager, ResourceManagerError
from .settings import Settings, get_settings
from .species_config import SpeciesConfig
from .utils import download_file, str_to_bgr
from .weights_manager_protocol import WeightsManagerProtocol

__all__ = [
    # Base classes and protocols
    "BasePredictor",
    "BaseProvider",
    "WeightsManagerProtocol",
    # Configuration
    "ConfigManager",
    "CulicidaeLabConfig",
    "PredictorConfig",
    "DatasetConfig",
    "ProviderConfig",
    "SpeciesModel",
    "SpeciesConfig",
    # Services and Managers
    "ProviderService",
    "ResourceManager",
    "ResourceManagerError",
    # Settings Facade
    "Settings",
    "get_settings",
    # Utilities
    "download_file",
    "str_to_bgr",
]
