"""
CulicidaeLab - A Python library for mosquito detection, segmentation, and classification
"""

from __future__ import annotations

from .core.config_manager import ConfigManager, ConfigurableComponent
from .core.resource_manager import ResourceManager
from .core.settings import Settings, get_settings
from .core.species_config import SpeciesConfig
from .core.utils import download_file, default_progress_callback

from .core.base_predictor import BasePredictor
from .predictors.classifier import MosquitoClassifier
from .predictors.detector import MosquitoDetector
from .predictors.segmenter import MosquitoSegmenter
from .predictors.model_weights_manager import ModelWeightsManager

from .datasets.datasets_manager import DatasetsManager
from .datasets.huggingface import HuggingFaceDatasetLoader

from .core.base_provider import BaseProvider
from .providers.huggingface_provider import HuggingFaceProvider
# from .providers.kaggle_provider import KaggleProvider
# from .providers.remote_url_provider import RemoteURLProvider
# from .providers.roboflow_provider import RoboflowProvider

__all__ = [
    'ConfigManager',
    'ConfigurableComponent',
    'ResourceManager',
    'Settings',
    'SpeciesConfig',
    'download_file',
    'default_progress_callback',

    'BasePredictor',
    'MosquitoClassifier',
    'MosquitoDetector',
    'MosquitoSegmenter',
    'ModelWeightsManager',

    'DatasetsManager',
    'HuggingFaceDatasetLoader',

    'BaseProvider',
    'HuggingFaceProvider',
    # 'KaggleProvider',
    # 'RemoteURLProvider',
    # 'RoboflowProvider',
]


def __getattr__(name):
    if name != "__version__":
        msg = f"module {__name__} has no attribute {name}"
        raise AttributeError(msg)
    from importlib.metadata import version

    return version("culicidaelab")
