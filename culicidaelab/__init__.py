"""
CulicidaeLab - A Python library for mosquito detection, segmentation, and classification
"""

from __future__ import annotations

__version__ = "0.1.0"

from .classification import MosquitoClassifier
from .detection import MosquitoDetector
from .segmentation import MosquitoSegmenter

__all__ = ["MosquitoDetector", "MosquitoSegmenter", "MosquitoClassifier"]
