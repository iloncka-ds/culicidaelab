from typing import Union, TypeAlias
from pathlib import Path
import numpy as np
from PIL import Image

from culicidaelab.core.settings import get_settings
from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.predictors.backend_factory import create_backend
from culicidaelab.predictors import MosquitoClassifier, MosquitoDetector, MosquitoSegmenter
from culicidaelab.core.prediction_models import (
    ClassificationPrediction,
    DetectionPrediction,
    SegmentationPrediction,
)

# Define the possible inputs and outputs for clarity
ImageInput: TypeAlias = Union[np.ndarray, str, Path, Image.Image, bytes]
PredictionResult: TypeAlias = Union[ClassificationPrediction, DetectionPrediction, SegmentationPrediction]

# In-memory cache to hold initialized predictor instances for performance
_PREDICTOR_CACHE: dict[str, BasePredictor] = {}


def serve(
    image: ImageInput,
    predictor_type: str = "classifier",
    **kwargs,
) -> PredictionResult:
    """
    High-level function for running production inference.

    This function automatically uses the lightweight, high-performance ONNX backend
    and caches predictor instances in memory for speed.

    Args:
        image: The input image (path, numpy array, bytes, etc.).
        predictor_type: The type of predictor to use ('classifier', 'detector', 'segmenter').
        **kwargs: Additional arguments to pass to the predictor's predict method
                  (e.g., confidence_threshold for a detector).

    Returns:
        A Pydantic model containing the structured prediction results.
    """
    if predictor_type not in _PREDICTOR_CACHE:
        settings = get_settings()
        predictor_class_map: dict[str, type[BasePredictor]] = {
            "classifier": MosquitoClassifier,
            "detector": MosquitoDetector,
            "segmenter": MosquitoSegmenter,
        }

        if predictor_type not in predictor_class_map:
            raise ValueError(
                f"Unknown predictor_type: '{predictor_type}'. "
                f"Available options are: {list(predictor_class_map.keys())}",
            )

        predictor_class = predictor_class_map[predictor_type]

        backend_instance = create_backend(
            predictor_type=predictor_type,
            settings=settings,
            mode="serve",
        )
        # Instantiate the predictor, FORCING the 'serve' mode to guarantee ONNX is used.
        # This overrides any local YAML configuration for maximum safety.
        print(f"Initializing '{predictor_type}' predictor for serving...")
        predictor = predictor_class(settings, predictor_type=predictor_type, backend=backend_instance)
        _PREDICTOR_CACHE[predictor_type] = predictor

    # Retrieve the cached predictor and run prediction
    predictor_instance = _PREDICTOR_CACHE[predictor_type]
    return predictor_instance.predict(image, **kwargs)


def clear_serve_cache():
    """Utility function to clear the in-memory predictor cache."""
    global _PREDICTOR_CACHE
    for predictor in _PREDICTOR_CACHE.values():
        predictor.unload_model()
    _PREDICTOR_CACHE = {}
