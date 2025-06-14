"""
Module for mosquito segmentation using SAM (Segment Anything Model).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path
from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr


class MosquitoSegmenter(BasePredictor):
    """Class for segmenting mosquitos in images using SAM."""

    def __init__(self, settings: Settings,
                load_model: bool = False) -> None:
        """
        Initialize the mosquito segmenter.

        Args:
            model_path: Path to SAM model checkpoint
            config_manager: Configuration manager instance
        """
        super().__init__(settings=settings,
                         predictor_type="segmenter",
                         load_model=load_model)

    def _load_model(self) -> None:
        """Load the SAM model."""

        sam2_model = build_sam2(self.config.sam_config_path,
                                self.model_path,
                                device=self.config.device)
        self._model = SAM2ImagePredictor(sam2_model)

    def predict(
        self,
        input_data: np.ndarray,
        detection_boxes: list[tuple[float, float, float, float, float]] | None = None,
    ) -> np.ndarray:
        """
        Segment mosquitos in an image.

        Args:
            input_data: Input image as numpy array
            detection_boxes: Optional list of detection boxes (x, y, w, h, conf)

        Returns:
            np.ndarray: Binary mask of segmented mosquitos
        """
        if not self.model_loaded:
            self.load_model()

        if len(input_data.shape) == 2:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
        elif input_data.shape[2] == 4:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)

        self._model.set_image(input_data)

        if detection_boxes:
            masks = []
            for box in detection_boxes:
                x, y, w, h, _ = box
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                input_box = np.array([x1, y1, x2, y2])
                mask, _, _ = self._model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                masks.append(mask)
            return np.logical_or.reduce(masks) if masks else np.zeros(input_data.shape[:2], dtype=bool)

        masks = self._model.generate()
        return (
            np.logical_or.reduce([m["segmentation"] for m in masks])
            if masks
            else np.zeros(input_data.shape[:2], dtype=bool)
        )

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: np.ndarray,
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
        overlay = input_data.copy()
        overlay[predictions] = cv2.addWeighted(
            overlay[predictions],
            self.config.visualization["alpha"],
            np.array(str_to_bgr(self.config.visualization["overlay_color"])),
            1 - self.config.visualization["alpha"],
            0,
        )

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay

    def evaluate(
        self,
        input_data: np.ndarray,
        ground_truth: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate segmentation predictions against ground truth mask.

        Args:
            input_data: Input image
            ground_truth: Ground truth binary mask

        Returns:
            Dictionary containing IoU and other metrics
        """
        predictions = self.predict(input_data)

        intersection = np.logical_and(predictions, ground_truth)
        union = np.logical_or(predictions, ground_truth)

        iou = float(np.sum(intersection)) / float(np.sum(union)) if np.sum(union) > 0 else 0.0
        precision = float(np.sum(intersection)) / float(np.sum(predictions)) if np.sum(predictions) > 0 else 0.0
        recall = float(np.sum(intersection)) / float(np.sum(ground_truth)) if np.sum(ground_truth) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
