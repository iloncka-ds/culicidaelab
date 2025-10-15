"""High-level inference functions for production environments.

This module provides a simplified, high-performance interface for running predictions.
It is designed for production use cases where speed and reliability are critical.
The `serve` function automatically utilizes the ONNX backend for inference, which
is optimized for lightweight and fast execution. To avoid the overhead of repeated
model loading, it caches predictor instances in memory.

Example:
    from culicidaelab.serve import serve, clear_serve_cache
    from PIL import Image

    # Run a prediction
    image_path = "path/to/your/image.jpg"
    prediction = serve(image_path, predictor_type="detector", confidence_threshold=0.5)
    print(prediction)

    # Clear the cache when done or to free up resources
    clear_serve_cache()
"""

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
"""Type alias for acceptable image input formats."""

PredictionResult: TypeAlias = Union[ClassificationPrediction, DetectionPrediction, SegmentationPrediction]
"""Type alias for the structured prediction output."""

# In-memory cache to hold initialized predictor instances for performance
_PREDICTOR_CACHE: dict[str, BasePredictor] = {}


def serve(
    image: ImageInput,
    predictor_type: str = "classifier",
    **kwargs,
) -> PredictionResult:
    """Runs a prediction using a specified predictor type in high-performance mode.

    This function is optimized for production environments. It automatically selects
    the ONNX backend for fast inference and caches the initialized predictor in
    memory to minimize latency on subsequent calls. The first call for a given
    `predictor_type` will be slower as it includes model initialization.

    Args:
        image: The input image to process. Can be a file path (str or Path),
            a PIL Image, a NumPy array, or bytes.
        predictor_type: The type of predictor to use. Valid options are
            'classifier', 'detector', and 'segmenter'. Defaults to 'classifier'.
        **kwargs: Additional keyword arguments to pass to the predictor's
            `predict` method. For example, `confidence_threshold=0.5` for
            a detector.

    Returns:
        A Pydantic model (`ClassificationPrediction`, `DetectionPrediction`, or
        `SegmentationPrediction`) containing the structured prediction results.

    Raises:
        ValueError: If an unknown `predictor_type` is specified.

    Example:
        >>> from culicidaelab.serve import serve
        >>> from PIL import Image
        >>>
        >>> # Create a dummy image
        >>> dummy_image = Image.new('RGB', (100, 100), color = 'red')
        >>>
        >>> # Run mosquito detection
        >>> detection_result = serve(dummy_image, predictor_type="detector", confidence_threshold=0.5)
        >>> print(detection_result.model_dump_json(indent=2))
        >>>
        >>> # Run mosquito classification
        >>> classification_result = serve(dummy_image, predictor_type="classifier")
        >>> print(classification_result.model_dump_json(indent=2))
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

        # Create the backend instance configured for serving (ONNX)
        backend_instance = create_backend(
            predictor_type=predictor_type,
            settings=settings,
            mode="serve",
        )
        # Instantiate the predictor, forcing the 'serve' mode to guarantee ONNX is used.
        # This overrides any local YAML configuration for maximum safety.
        print(f"Initializing '{predictor_type}' predictor for serving...")
        predictor = predictor_class(settings, predictor_type=predictor_type, backend=backend_instance)
        _PREDICTOR_CACHE[predictor_type] = predictor

    # Retrieve the cached predictor and run prediction
    predictor_instance = _PREDICTOR_CACHE[predictor_type]
    return predictor_instance.predict(image, **kwargs)


def clear_serve_cache():
    """Clears the in-memory predictor cache and unloads models.

    This function should be called to release GPU memory and other resources
    when the `serve` function is no longer needed. It iterates through all
    cached predictors, unloads their models, and then clears the cache dictionary.
    """
    global _PREDICTOR_CACHE
    for predictor in _PREDICTOR_CACHE.values():
        # Ensure the model and any associated resources are released
        predictor.unload_model()
    _PREDICTOR_CACHE = {}
