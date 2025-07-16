"""Module for mosquito segmentation using the Segment Anything Model (SAM).

This module provides the MosquitoSegmenter class, which uses a pre-trained
SAM model (specifically, SAM2) to generate precise segmentation masks for
mosquitos in an image. It can be prompted with detection bounding boxes
for targeted segmentation.

Example:
    from culicidaelab.core.settings import Settings
    from culicidaelab.predictors import MosquitoSegmenter
    import numpy as np

    # Initialize settings and segmenter
    settings = Settings()
    segmenter = MosquitoSegmenter(settings, load_model=True)

    # Create a dummy image and a detection box
    image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    # box = (center_x, center_y, width, height, confidence)
    detection_box = [(512, 512, 100, 100, 0.9)]

    # Get segmentation mask
    mask = segmenter.predict(image, detection_boxes=detection_box)

    print(f"Generated a mask of shape: {mask.shape}")
    print(f"Number of segmented pixels: {np.sum(mask)}")

    # Clean up
    segmenter.unload_model()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias, cast

import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

SegmentationPredictionType: TypeAlias = np.ndarray
SegmentationGroundTruthType: TypeAlias = np.ndarray


class MosquitoSegmenter(BasePredictor[SegmentationPredictionType, SegmentationGroundTruthType]):
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

    def predict(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generates a segmentation mask for mosquitos in an image.

        If `detection_boxes` are provided in kwargs, the model will generate
        masks only for those specific regions. Otherwise, it will attempt to
        segment the most prominent object in the image.

        Args:
            input_data (np.ndarray): The input image as a NumPy array (H, W, 3).
            **kwargs (Any): Optional keyword arguments, including:
                detection_boxes (list, optional): A list of bounding boxes
                in (cx, cy, w, h, conf) format to guide segmentation.

        Returns:
            np.ndarray: A binary segmentation mask of shape (H, W), where pixels
            belonging to a mosquito are marked as 1 (or True) and 0 otherwise.

        Raises:
            RuntimeError: If the model is not loaded or fails to load.
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

        masks, _, _ = model.predict(point_coords=None, point_labels=None, box=None, multimask_output=False)
        return masks[0].astype(np.uint8)

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: SegmentationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Overlays a segmentation mask on the original image.

        Args:
            input_data (np.ndarray): The original image.
            predictions (SegmentationPredictionType): The binary segmentation mask
                from the `predict` method.
            save_path (str | Path | None, optional): If provided, the visualized
                image is saved to this path. Defaults to None.

        Returns:
            np.ndarray: A new image array with the segmentation mask visualized
            as a colored overlay.
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
        """Calculates segmentation metrics for a single predicted mask.

        Computes Intersection over Union (IoU), precision, recall, and F1-score.

        Args:
            prediction (SegmentationPredictionType): The predicted binary mask.
            ground_truth (SegmentationGroundTruthType): The ground truth binary mask.

        Returns:
            dict[str, float]: A dictionary containing the segmentation metrics.
        """
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

        return {"iou": float(iou), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    def _load_model(self) -> None:
        """Loads the SAM2 model and initializes the image predictor.

        Raises:
            ValueError: If the required configuration for the model path or
                device is missing.
            RuntimeError: If the model fails to load from the specified path.
        """
        if (
            not hasattr(self.config, "model_config_path")
            or self.config.model_config_path is None
            or not hasattr(self.config, "device")
        ):
            raise ValueError("Missing required configuration: 'model_config_path' and 'device' must be set")

        sam2_model = build_sam2(self.config.model_config_path, str(self.model_path), device=self.config.device)
        try:
            self._model = SAM2ImagePredictor(sam2_model)
            self._model_loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAM model from {self.model_path}. "
                f"Please check the model path and configuration. Error: {str(e)}",
            )
