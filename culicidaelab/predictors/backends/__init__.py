"""This package contains the various backend implementations for the predictors."""

from .classifier._fastai import ClassifierFastAIBackend
from .classifier._onnx import ClassifierONNXBackend
from .detector._yolo import DetectorYOLOBackend
from .segmenter._sam import SegmenterSAMBackend

__all__ = [
    "ClassifierFastAIBackend",
    "ClassifierONNXBackend",
    "DetectorYOLOBackend",
    "SegmenterSAMBackend",
]
