"""Module for mosquito segmentation using the Segment Anything Model (SAM).

This module provides the MosquitoSegmenter class, which uses a pre-trained
SAM model (specifically, SAM2) to generate precise segmentation masks for
mosquitos in an image. It can be prompted with detection bounding boxes
for targeted segmentation and supports efficient batch processing.

Example:
    from culicidaelab.core.settings import Settings
    from culicidaelab.predictors import MosquitoSegmenter
    import numpy as np

    # Initialize settings and segmenter
    settings = Settings()
    segmenter = MosquitoSegmenter(settings, load_model=True)

    # --- Batch Prediction Example ---
    # Create a dummy batch of images and detection boxes
    images = [np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8) for _ in range(4)]
    # box = (x1, y1, x2, y2, confidence)
    detections_batch = [
        [(512, 512, 100, 100, 0.9)],  # Boxes for image 1
        [(256, 256, 80, 80, 0.95), (700, 700, 120, 120, 0.88)],  # Boxes for image 2
        [],  # No boxes for image 3
        [(100, 800, 50, 50, 0.92)],  # Boxes for image 4
    ]

    # Get segmentation masks for the whole batch
    masks = segmenter.predict_batch(images, detection_boxes_batch=detections_batch)

    print(f"Generated {len(masks)} masks for the batch.")
    print(f"Shape of first mask: {masks[0].shape}")

    # Clean up
    segmenter.unload_model()
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image  # Added Image


from culicidaelab.core.base_predictor import BasePredictor, ImageInput
from culicidaelab.core.prediction_models import SegmentationPrediction
from culicidaelab.core.settings import Settings
from culicidaelab.predictors.backend_factory import create_backend

SegmentationGroundTruthType: TypeAlias = np.ndarray


class MosquitoSegmenter(
    BasePredictor[ImageInput, SegmentationPrediction, SegmentationGroundTruthType],
):
    """Segments mosquitos in images using the SAM2 model.

    This class provides methods to load a SAM2 model, generate segmentation
    masks for entire images or specific regions defined by bounding boxes,
    and visualize the resulting masks.

    Args:
        settings (Settings): The main settings object for the library.
        load_model (bool, optional): If True, the model is loaded upon
            initialization. Defaults to False.
    """

    def __init__(self, settings: Settings, load_model: bool = False, backend: str | None = None) -> None:
        """Initializes the MosquitoSegmenter."""
        predictor_type = "segmenter"
        config = settings.get_config(f"predictors.{predictor_type}")
        backend_name = backend or config.backend or "torch"

        backend_instance = create_backend(settings, predictor_type, backend_name)

        super().__init__(
            settings=settings,
            predictor_type=predictor_type,
            backend=backend_instance,
            load_model=load_model,
        )

    def _convert_raw_to_prediction(self, raw_prediction: np.ndarray) -> SegmentationPrediction:
        """ """
        return SegmentationPrediction(mask=raw_prediction, pixel_count=int(np.sum(raw_prediction)))

    def visualize(
        self,
        input_data: ImageInput,
        predictions: SegmentationPrediction,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Overlays a segmentation mask on the original image."""

        image_pil = self._load_and_validate_image(input_data)

        colored_mask = Image.new("RGB", image_pil.size, self.config.visualization.overlay_color)

        alpha_mask = Image.fromarray(((1 - predictions.mask) * 255).astype(np.uint8))
        overlay = Image.composite(image_pil, colored_mask, alpha_mask)

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
        """Calculates segmentation metrics for a single predicted mask."""
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
