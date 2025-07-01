"""
Module for mosquito segmentation using SAM (Segment Anything Model).
"""

from __future__ import annotations

from typing import Any, TypeAlias, cast
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path
from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.utils import str_to_bgr
from .model_weights_manager import ModelWeightsManager


SegmentationPredictionType: TypeAlias = np.ndarray
SegmentationGroundTruthType: TypeAlias = np.ndarray


class MosquitoSegmenter(BasePredictor[SegmentationPredictionType, SegmentationGroundTruthType]):
    """Class for segmenting mosquitos in images using SAM."""

    def __init__(
        self,
        settings: Settings,
        load_model: bool = False,
    ) -> None:
        """
        Initialize the mosquito segmenter.

        Args:
            settings: The main Settings object for the library.
            load_model: If True, loads model immediately.
        """
        provider_service = ProviderService(settings)
        weights_manager = ModelWeightsManager(
            settings=settings,
            provider_service=provider_service,
        )
        super().__init__(
            settings=settings,
            predictor_type="segmenter",
            weights_manager=weights_manager,
            load_model=load_model,
        )

    def _load_model(self) -> None:
        """Load the SAM model."""
        sam2_model = None
        if (
            not hasattr(self.config, "model_config_path")
            or self.config.model_config_path is None
            or not hasattr(self.config, "device")
        ):
            raise ValueError("Missing required configuration: 'sam_config_path' and 'device' must be set")

        sam_config_path = str(self.settings.model_dir / self.config.model_config_path)
        sam2_model = build_sam2(sam_config_path, str(self.model_path), device=self.config.device)
        try:
            self._model = SAM2ImagePredictor(sam2_model)
            self._model_loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAM model from {self.model_path}. "
                f"Please check the model path and configuration. Error: {str(e)}",
            )

    def predict(
        self,
        input_data: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Segment mosquitos in an image.

        Args:
            input_data: Input image as numpy array
            **kwargs: Additional arguments including:
                detection_boxes: Optional list of detection boxes (x, y, w, h, conf)

        Returns:
            np.ndarray: Binary mask of segmented mosquitos
        """
        if not self.model_loaded or self._model is None:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Failed to load model")

        detection_boxes = kwargs.get("detection_boxes")

        if len(input_data.shape) == 2:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
        elif input_data.shape[2] == 4:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)

        model = cast(SAM2ImagePredictor, self._model)
        model.set_image(input_data)

        if detection_boxes and len(detection_boxes) > 0:
            masks = []
            for box in detection_boxes:
                x, y, w, h, _ = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                input_box = np.array([x1, y1, x2, y2])

                mask, _, _ = model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                masks.append(mask[0].astype(np.uint8))
            return np.logical_or.reduce(masks) if masks else np.zeros(input_data.shape[:2], dtype=bool)

        masks, scores, _ = model.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=False,
        )

        return masks[0].astype(np.uint8)

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: SegmentationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Visualize segmentation mask on the image.

        Args:
            input_data: Original image
            predictions: Binary segmentation mask
            save_path: Optional path to save visualization

        Returns:
            np.ndarray: Image with overlay visualization
        """
        if len(input_data.shape) == 2:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR)

        colored_mask = np.zeros_like(input_data)
        overlay_color_bgr = str_to_bgr(self.config.visualization.overlay_color)
        colored_mask[predictions > 0] = np.array(overlay_color_bgr)

        overlay = cv2.addWeighted(input_data, 1, colored_mask, self.config.visualization.alpha, 0)

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay

    def _evaluate_from_prediction(
        self,
        prediction: SegmentationPredictionType,
        ground_truth: SegmentationGroundTruthType,
    ) -> dict[str, float]:
        """
        The core logic for calculating segmentation metrics for a single item.

        Args:
            prediction: The predicted binary mask from the model.
            ground_truth: The ground truth binary mask.

        Returns:
            Dictionary containing IoU, precision, recall, and F1-score.
        """
        # Ensure boolean arrays for logical operations, which is more robust
        prediction = prediction.astype(bool)
        ground_truth = ground_truth.astype(bool)

        intersection = np.logical_and(prediction, ground_truth)
        union = np.logical_or(prediction, ground_truth)

        intersection_sum = np.sum(intersection)
        prediction_sum = np.sum(prediction)
        ground_truth_sum = np.sum(ground_truth)
        union_sum = np.sum(union)

        iou = intersection_sum / union_sum if union_sum > 0 else 0.0
        precision = intersection_sum / prediction_sum if prediction_sum > 0 else 0.0
        recall = intersection_sum / ground_truth_sum if ground_truth_sum > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
