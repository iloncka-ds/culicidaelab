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
from typing import Any, TypeAlias, cast
from collections.abc import Sequence

import torch
import cv2
import numpy as np
from ultralytics import SAM
from fastprogress.fastprogress import progress_bar


from culicidaelab.core.base_predictor import BasePredictor, ImageInput

from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

SegmentationPredictionType: TypeAlias = np.ndarray
SegmentationGroundTruthType: TypeAlias = np.ndarray


class MosquitoSegmenter(
    BasePredictor[ImageInput, SegmentationPredictionType, SegmentationGroundTruthType],
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

    def __init__(self, settings: Settings, load_model: bool = False) -> None:
        """Initializes the MosquitoSegmenter."""

        weights_manager = ModelWeightsManager(
            settings=settings,
        )
        super().__init__(
            settings=settings,
            predictor_type="segmenter",
            weights_manager=weights_manager,
            load_model=load_model,
        )

    def predict(self, input_data: ImageInput, **kwargs: Any) -> np.ndarray:
        if not self.model_loaded:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Model could not be loaded for prediction.")
        model = cast(SAM, self._model)
        detection_boxes = kwargs.get("detection_boxes")
        image_pil = self._load_and_validate_image(input_data)
        image_np = np.array(image_pil)
        h, w, _ = image_np.shape

        is_empty = False
        if detection_boxes is None:
            is_empty = True
        elif hasattr(detection_boxes, "size"):
            if detection_boxes.size == 0:
                is_empty = True
        elif not detection_boxes:
            is_empty = True

        if is_empty:
            message = "No detection boxes provided; returning empty mask."
            self._logger.debug(message)
            print(message)
            return np.zeros((h, w), dtype=np.uint8)
        else:
            self._logger.debug(f"Using {len(detection_boxes)} detection boxes for segmentation.")

            boxes_xyxy = []
            first_box = detection_boxes[0]
            if len(first_box) == 5:
                boxes_xyxy = [box[:4] for box in detection_boxes]
            elif len(first_box) == 4:
                boxes_xyxy = detection_boxes
            else:
                self._logger.warning("Invalid format for detection_boxes.")
                return np.zeros((h, w), dtype=np.uint8)

            self._logger.debug(f"Using {len(boxes_xyxy)} detection boxes for segmentation.")

            results = model(image_np, bboxes=boxes_xyxy, verbose=False)
            result = results[0]

        if result.masks is None:
            return np.zeros((h, w), dtype=np.uint8)

        masks_np = self._to_numpy(result.masks.data)
        return (
            np.logical_or.reduce(masks_np).astype(np.uint8)
            if masks_np.shape[0] > 0
            else np.zeros((h, w), dtype=np.uint8)
        )

    def predict_batch(
        self,
        input_data_batch: Sequence[ImageInput],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[SegmentationPredictionType]:
        """Generates segmentation masks for a batch of images by serial processing."""
        # This method remains unchanged from the previous correct version.
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        detection_boxes_batch = kwargs.get("detection_boxes_batch", [[] for _ in input_data_batch])

        if len(input_data_batch) != len(detection_boxes_batch):
            raise ValueError(
                f"Mismatch between number of images ({len(input_data_batch)}) and "
                f"number of detection box lists ({len(detection_boxes_batch)}).",
            )

        final_masks: list[SegmentationPredictionType] = []
        iterator = enumerate(input_data_batch)
        if show_progress:
            iterator = progress_bar(iterator, total=len(input_data_batch))

        for i, input_data in iterator:
            try:
                mask = self.predict(input_data, detection_boxes=detection_boxes_batch[i])
                final_masks.append(mask)
            except Exception as e:
                self._logger.error(f"Failed to process image at index {i}: {e}")
                final_masks.append(np.zeros((1, 1), dtype=np.uint8))  # Append placeholder
        return final_masks

    def visualize(
        self,
        input_data: ImageInput,
        predictions: SegmentationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Overlays a segmentation mask on the original image."""

        image_np = np.array(self._load_and_validate_image(input_data))

        colored_mask = np.zeros_like(image_np)
        overlay_color_bgr = str_to_bgr(self.config.visualization.overlay_color)

        # This can fail if shapes mismatch, which is expected behavior
        colored_mask[predictions > 0] = np.array(overlay_color_bgr)

        overlay = cv2.addWeighted(
            image_np,
            1,
            colored_mask,
            self.config.visualization.alpha,
            0,
        )

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay

    def _evaluate_from_prediction(
        self,
        prediction: SegmentationPredictionType,
        ground_truth: SegmentationGroundTruthType,
    ) -> dict[str, float]:
        """Calculates segmentation metrics for a single predicted mask."""
        prediction = prediction.astype(bool)
        ground_truth = ground_truth.astype(bool)

        if prediction.shape != ground_truth.shape:
            raise ValueError("Prediction and ground truth must have the same shape.")

        intersection = np.logical_and(prediction, ground_truth).sum()
        union = np.logical_or(prediction, ground_truth).sum()
        prediction_sum = prediction.sum()
        ground_truth_sum = ground_truth.sum()

        iou = intersection / union if union > 0 else 0.0
        precision = intersection / prediction_sum if prediction_sum > 0 else 0.0
        recall = intersection / ground_truth_sum if ground_truth_sum > 0 else 0.0
        f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {"iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    def _load_model(self) -> None:
        """Loads the SAM2 model using Ultralytics, selecting the best available device."""
        device_str = self.config.device or "auto"
        device = torch.device("cuda" if torch.cuda.is_available() and device_str == "auto" else device_str)
        self._logger.info(f"Using device: {device}")

        try:
            model_path_str = self.model_path.as_posix()
            self._logger.info(f"Loading SAM model from: {model_path_str}")
            model = SAM(model_path_str)
            model.to(device)
            self._model = model
        except Exception as e:
            # The base class's load_model will handle setting _model_loaded and raising
            raise RuntimeError(f"Failed to load SAM model. Error: {e}") from e

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """Safely converts a torch.Tensor to a numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, "detach"):
            return tensor.detach().cpu().numpy()
        return np.array(tensor)
