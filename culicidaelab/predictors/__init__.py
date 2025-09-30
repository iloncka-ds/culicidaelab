"""This package contains the predictor classes for the culicidaelab library."""

from .classifier import MosquitoClassifier
from .detector import MosquitoDetector
from .segmenter import MosquitoSegmenter
from .model_weights_manager import ModelWeightsManager

__all__ = [
    "MosquitoClassifier",
    "MosquitoDetector",
    "MosquitoSegmenter",
    "ModelWeightsManager",
]
