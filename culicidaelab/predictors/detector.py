"""Module for mosquito object detection in images using YOLO.

This module provides the MosquitoDetector class, which uses a pre-trained
YOLO model from the `ultralytics` library to find bounding boxes of
mosquitos in an image.

Example:
    from culicidaelab.core.settings import Settings
    from culicidaelab.predictors import MosquitoDetector
    import numpy as np

    # Initialize settings and detector
    settings = Settings()
    detector = MosquitoDetector(settings, load_model=True)

    # Create a dummy image
    image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

    # Get detections
    # Each detection is (center_x, center_y, width, height, confidence)
    detections = detector.predict(image)
    if detections:
        print(f"Found {len(detections)} mosquitos.")
        print(f"Top detection box: {detections[0]}")

    # Clean up
    detector.unload_model()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import ImageDraw, ImageFont

from culicidaelab.core.base_predictor import BasePredictor, ImageInput
from culicidaelab.core.prediction_models import (
    BoundingBox,
    Detection,
    DetectionPrediction,
)
from culicidaelab.core.settings import Settings
from culicidaelab.predictors.backend_factory import create_backend

DetectionGroundTruthType = list[tuple[float, float, float, float]]

logger = logging.getLogger(__name__)


class MosquitoDetector(
    BasePredictor[ImageInput, DetectionPrediction, DetectionGroundTruthType],
):
    """Detects mosquitos in images using a YOLO model.

    This class loads a YOLO model and provides methods for predicting bounding
    boxes on single or batches of images, visualizing results, and evaluating
    detection performance against ground truth data.

    Args:
        settings (Settings): The main settings object for the library.
        load_model (bool, optional): If True, the model is loaded upon
            initialization. Defaults to False.

    Attributes:
        confidence_threshold (float): The minimum confidence score for a
            detection to be considered valid.
        iou_threshold (float): The IoU threshold for non-maximum suppression.
        max_detections (int): The maximum number of detections to return per image.
    """

    def __init__(self, settings: Settings, load_model: bool = False, backend: str | None = None) -> None:
        """Initializes the MosquitoDetector."""
        predictor_type = "detector"
        config = settings.get_config(f"predictors.{predictor_type}")
        backend_name = backend or config.backend or "torch"

        backend_instance = create_backend(settings, predictor_type, backend_name)

        super().__init__(
            settings=settings,
            predictor_type=predictor_type,
            backend=backend_instance,
            load_model=load_model,
        )
        self.confidence_threshold: float = self.config.confidence or 0.5
        self.iou_threshold: float = self.config.params.get("iou_threshold", 0.45)
        self.max_detections: int = self.config.params.get("max_detections", 300)

    def predict(self, input_data: ImageInput, **kwargs: Any) -> DetectionPrediction:
        """Detects mosquitos in a single image.

        Args:
            input_data (ImageInput): The input image as a NumPy array or other supported format.
            **kwargs (Any): Optional keyword arguments, including:
                confidence_threshold (float): Override the default confidence
                    threshold for this prediction.

        Returns:
            DetectionPrediction: A `DetectionPrediction` object containing a list of
            `Detection` instances. Returns an empty list if no mosquitos are found.

        Raises:
            RuntimeError: If the model fails to load or if prediction fails.
        """
        if not self.backend.is_loaded:
            self.load_model()

        confidence_threshold = kwargs.get(
            "confidence_threshold",
            self.confidence_threshold,
        )

        try:
            input_image = np.array(self._load_and_validate_image(input_data))
            # The backend now returns a standardized NumPy array (N, 5) -> [x1, y1, x2, y2, conf]
            results_array = self.backend.predict(
                input_data=input_image,
                conf=confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}") from e

        return self._convert_raw_to_prediction(results_array)

    def _convert_raw_to_prediction(self, raw_prediction: np.ndarray) -> DetectionPrediction:
        """ """
        detections: list[Detection] = []
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 5:
            for row in raw_prediction:
                x1, y1, x2, y2, conf = row
                detections.append(
                    Detection(box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), confidence=conf),
                )
        return DetectionPrediction(detections=detections)

    def visualize(
        self,
        input_data: ImageInput,
        predictions: DetectionPrediction,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Draws predicted bounding boxes on an image.

        Args:
            input_data (ImageInput): The original image.
            predictions (DetectionPrediction): The `DetectionPrediction` from `predict`.
            save_path (str | Path | None, optional): If provided, the output
                image is saved to this path. Defaults to None.

        Returns:
            np.ndarray: A new image array with bounding boxes and confidence
            scores drawn on it.
        """
        vis_img = self._load_and_validate_image(input_data).copy()
        draw = ImageDraw.Draw(vis_img)
        vis_config = self.config.visualization
        # box_color = str_to_bgr(vis_config.box_color) # Removed
        # text_color = str_to_bgr(vis_config.text_color) # Removed
        font_scale = vis_config.font_scale
        thickness = vis_config.box_thickness

        for detection in predictions.detections:
            box = detection.box
            conf = detection.confidence
            draw.rectangle(
                [(int(box.x1), int(box.y1)), (int(box.x2), int(box.y2))],
                outline=vis_config.box_color,
                width=thickness,
            )
            text = f"{conf:.2f}"
            # Load a font (you might want to make this configurable or load once)
            try:
                font = ImageFont.truetype("arial.ttf", int(font_scale * 20))  # Adjust font size as needed
            except OSError:
                font = ImageFont.load_default()
            draw.text((int(box.x1), int(box.y1 - 10)), text, fill=vis_config.text_color, font=font)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            vis_img.save(str(save_path))

        return np.array(vis_img)

    def _calculate_iou(self, box1_xyxy: tuple, box2_xyxy: tuple) -> float:
        """Calculates Intersection over Union (IoU) for two boxes.

        Args:
            box1_xyxy (tuple): The first box in (x1, y1, x2, y2) format.
            box2_xyxy (tuple): The second box in (x1, y1, x2, y2) format.

        Returns:
            float: The IoU score between 0.0 and 1.0.
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy
        b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy

        inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
        inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _evaluate_from_prediction(
        self,
        prediction: DetectionPrediction,
        ground_truth: DetectionGroundTruthType,
    ) -> dict[str, float]:
        """Calculates detection metrics for a single image's predictions.

        This computes precision, recall, F1-score, Average Precision (AP),
        and mean IoU for a set of predicted boxes against ground truth boxes.

        Args:
            prediction (DetectionPrediction): A `DetectionPrediction` object.
            ground_truth (DetectionGroundTruthType): A list of ground truth
                boxes: `[(x, y, w, h), ...]`.

        Returns:
            dict[str, float]: A dictionary containing the calculated metrics.
        """
        if not ground_truth and not prediction.detections:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "ap": 1.0,
                "mean_iou": 0.0,
            }
        if not ground_truth:  # False positives exist
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "mean_iou": 0.0,
            }
        if not prediction.detections:  # False negatives exist
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "mean_iou": 0.0,
            }

        predictions_sorted = sorted(prediction.detections, key=lambda x: x.confidence, reverse=True)
        tp = np.zeros(len(predictions_sorted))
        fp = np.zeros(len(predictions_sorted))
        gt_matched = [False] * len(ground_truth)
        all_ious_for_mean = []
        iou_threshold = self.iou_threshold

        for i, pred in enumerate(predictions_sorted):
            pred_box = (pred.box.x1, pred.box.y1, pred.box.x2, pred.box.y2)
            best_iou, best_gt_idx = 0.0, -1

            for j, gt_box in enumerate(ground_truth):
                if not gt_matched[j]:
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_gt_idx != -1:
                all_ious_for_mean.append(best_iou)

            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:  # Matched a GT box that was already matched
                    fp[i] = 1
            else:
                fp[i] = 1

        mean_iou_val = float(np.mean(all_ious_for_mean)) if all_ious_for_mean else 0.0
        fp_cumsum, tp_cumsum = np.cumsum(fp), np.cumsum(tp)
        recall_curve = tp_cumsum / len(ground_truth)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

        ap = 0.0
        for t in np.linspace(0, 1, 11):  # 11-point interpolation
            precisions_at_recall_t = precision_curve[recall_curve >= t]
            ap += np.max(precisions_at_recall_t) if len(precisions_at_recall_t) > 0 else 0.0
        ap /= 11.0

        final_precision = precision_curve[-1] if len(precision_curve) > 0 else 0.0
        final_recall = recall_curve[-1] if len(recall_curve) > 0 else 0.0
        f1 = (
            2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-9)
            if (final_precision + final_recall) > 0
            else 0.0
        )

        return {
            "precision": float(final_precision),
            "recall": float(final_recall),
            "f1": float(f1),
            "ap": float(ap),
            "mean_iou": mean_iou_val,
        }
