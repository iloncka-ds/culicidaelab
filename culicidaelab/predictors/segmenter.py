"""Module for mosquito segmentation using the Segment Anything Model (SAM).

This module provides the MosquitoSegmenter class, which uses a pre-trained
SAM model to generate precise segmentation masks for mosquitos in an image.
It can be prompted with detection bounding boxes for targeted segmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, Literal

import numpy as np
from PIL import Image

from culicidaelab.core.base_predictor import BasePredictor, ImageInput
from culicidaelab.core.prediction_models import SegmentationPrediction
from culicidaelab.core.settings import Settings
from culicidaelab.predictors.backend_factory import create_backend
from culicidaelab.core.base_inference_backend import BaseInferenceBackend

SegmentationGroundTruthType: TypeAlias = np.ndarray


class MosquitoSegmenter(
    BasePredictor[ImageInput, SegmentationPrediction, SegmentationGroundTruthType],
):
    """Segments mosquitos in images using a SAM model.

    This class provides methods to load a SAM model, generate segmentation
    masks for entire images or specific regions defined by bounding boxes,
    and visualize the resulting masks.

    Example:
        >>> from culicidaelab.core.settings import Settings
        >>> from culicidaelab.predictors import MosquitoSegmenter
        >>> import numpy as np
        >>> # This example assumes you have a configured settings object
        >>> settings = Settings()
        >>> segmenter = MosquitoSegmenter(settings, load_model=True)
        >>> image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        >>> # Predict without prompts (might not be effective for all backends)
        >>> prediction = segmenter.predict(image)
        >>> print(f"Generated mask with {prediction.pixel_count} pixels.")

    """

    def __init__(
        self,
        settings: Settings,
        predictor_type="segmenter",
        mode: Literal["torch", "serve"] | None = None,
        load_model: bool = False,
        backend: BaseInferenceBackend | None = None,
    ) -> None:
        """Initializes the MosquitoSegmenter.

        Args:
            settings: The main settings object for the library.
            predictor_type: The type of predictor. Defaults to "segmenter".
            mode: The mode to run the predictor in, 'torch' or 'serve'.
                If None, it's determined by the environment.
            load_model: If True, load the model upon initialization.
            backend: An optional backend instance. If not provided, one will be
                created based on the mode and settings.
        """

        backend_instance = backend or create_backend(
            predictor_type=predictor_type,
            settings=settings,
            mode=mode,
        )

        super().__init__(
            settings=settings,
            predictor_type=predictor_type,
            backend=backend_instance,
            load_model=load_model,
        )

    def _convert_raw_to_prediction(self, raw_prediction: np.ndarray) -> SegmentationPrediction:
        """Converts a raw numpy mask to a structured segmentation prediction.

        Args:
            raw_prediction: A 2D numpy array representing the segmentation mask.

        Returns:
            A SegmentationPrediction object containing the mask and pixel count.
        """
        return SegmentationPrediction(mask=raw_prediction, pixel_count=int(np.sum(raw_prediction)))

    def visualize(
        self,
        input_data: ImageInput,
        predictions: SegmentationPrediction,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Overlays a segmentation mask on the original image.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.predictors import MosquitoSegmenter
            >>> # This example assumes you have a configured settings object
            >>> settings = Settings()
            >>> segmenter = MosquitoSegmenter(settings, load_model=True)
            >>> image = "path/to/your/image.jpg"
            >>> # Assuming you have a prediction from segmenter.predict()
            >>> prediction = segmenter.predict(image)
            >>> viz_image = segmenter.visualize(image, prediction, save_path="viz.jpg")

        Args:
            input_data: The original image.
            predictions: The `SegmentationPrediction` from `predict`.
            save_path: If provided, the output image is saved to this path.

        Returns:
            A numpy array of the image with the segmentation mask overlaid.
        """

        image_pil = self._load_and_validate_image(input_data)

        colored_mask = Image.new("RGB", image_pil.size, self.config.visualization.overlay_color)

        # Create an alpha mask where the segmentation is transparent
        alpha_mask = Image.fromarray((predictions.mask * 255).astype(np.uint8))

        # Composite the images
        overlay = Image.composite(colored_mask, image_pil, alpha_mask)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(str(save_path))

        return np.array(overlay)

    def _evaluate_from_prediction(
        self,
        prediction: SegmentationPrediction,
        ground_truth: SegmentationGroundTruthType,
    ) -> dict[str, float]:
        """Calculates segmentation metrics for a single predicted mask.

        Computes Intersection over Union (IoU), precision, recall, and F1-score.

        Args:
            prediction: The `SegmentationPrediction` object.
            ground_truth: A 2D numpy array of the ground truth mask.

        Returns:
            A dictionary containing the calculated metrics.

        Raises:
            ValueError: If prediction and ground truth masks have different shapes.
        """
        pred_mask = prediction.mask.astype(bool)
        ground_truth = ground_truth.astype(bool)

        if pred_mask.shape != ground_truth.shape:
            raise ValueError("Prediction and ground truth must have the same shape.")

        intersection = np.logical_and(pred_mask, ground_truth).sum()
        union = np.logical_or(pred_mask, ground_truth).sum()
        prediction_sum = pred_mask.sum()
        ground_truth_sum = ground_truth.sum()

        iou = intersection / union if union > 0 else 0.0
        precision = intersection / prediction_sum if prediction_sum > 0 else 0.0
        recall = intersection / ground_truth_sum if ground_truth_sum > 0 else 0.0
        f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {"iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1)}
