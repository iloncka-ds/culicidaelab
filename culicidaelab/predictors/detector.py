"""Module for mosquito object detection in images.

This module provides the MosquitoDetector class, which uses a pre-trained
model (e.g., YOLO) to find bounding boxes of mosquitos in an image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

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
from culicidaelab.core.base_inference_backend import BaseInferenceBackend

DetectionGroundTruthType = list[tuple[float, float, float, float]]

logger = logging.getLogger(__name__)


class MosquitoDetector(
    BasePredictor[ImageInput, DetectionPrediction, DetectionGroundTruthType],
):
    """Detects mosquitos in images using a YOLO model.

    This class loads a model and provides methods for predicting bounding
    boxes on single or batches of images, visualizing results, and evaluating
    detection performance against ground truth data.

    Attributes:
        confidence_threshold (float): The minimum confidence score for a
            detection to be considered valid.
        iou_threshold (float): The IoU threshold for non-maximum suppression.
        max_detections (int): The maximum number of detections to return per image.
    """

    def __init__(
        self,
        settings: Settings,
        predictor_type="detector",
        mode: Literal["torch", "serve"] | None = None,
        load_model: bool = False,
        backend: BaseInferenceBackend | None = None,
    ) -> None:
        """Initializes the MosquitoDetector.

        Args:
            settings: The main settings object for the library.
            predictor_type: The type of predictor. Defaults to "detector".
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
        self.confidence_threshold: float = self.config.confidence or 0.5
        self.iou_threshold: float = self.config.params.get("iou_threshold", 0.45)
        self.max_detections: int = self.config.params.get("max_detections", 300)

    def predict(self, input_data: ImageInput, **kwargs: Any) -> DetectionPrediction:
        """Detects mosquitos in a single image.

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.predictors import MosquitoDetector
            >>> # This example assumes you have a configured settings object
            >>> settings = Settings()
            >>> detector = MosquitoDetector(settings, load_model=True)
            >>> image = "path/to/your/image.jpg"
            >>> detections = detector.predict(image)
            >>> for detection in detections.detections:
            ...     print(detection.box, detection.confidence)

        Args:
            input_data: The input image as a NumPy array or other supported format.
            **kwargs: Optional keyword arguments, including:
                confidence_threshold (float): Override the default confidence
                    threshold for this prediction.

        Returns:
            A `DetectionPrediction` object containing a list of
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
            input_image = self._load_and_validate_image(input_data)
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
        """Converts raw model output to a structured detection prediction.

        Args:
            raw_prediction: A numpy array with shape (N, 5) where each row is
                [x1, y1, x2, y2, confidence].

        Returns:
            A DetectionPrediction object containing a list of Detection objects.
        """
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

        Example:
            >>> from culicidaelab.settings import Settings
            >>> from culicidaelab.predictors import MosquitoDetector
            >>> # This example assumes you have a configured settings object
            >>> settings = Settings()
            >>> detector = MosquitoDetector(settings, load_model=True)
            >>> image = "path/to/your/image.jpg"
            >>> detections = detector.predict(image)
            >>> viz_image = detector.visualize(image, detections, save_path="viz.jpg")

        Args:
            input_data: The original image.
            predictions: The `DetectionPrediction` from `predict`.
            save_path: If provided, the output image is saved to this path.

        Returns:
            A new image array with bounding boxes and confidence scores drawn on it.
        """
        vis_img = self._load_and_validate_image(input_data).copy()
        draw = ImageDraw.Draw(vis_img)
        vis_config = self.config.visualization
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
            try:
                font = ImageFont.truetype("arial.ttf", int(font_scale * 20))
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
            box1_xyxy: The first box in (x1, y1, x2, y2) format.
            box2_xyxy: The second box in (x1, y1, x2, y2) format.

        Returns:
            The IoU score between 0.0 and 1.0.
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
            prediction: A `DetectionPrediction` object.
            ground_truth: A list of ground truth boxes: `[(x, y, w, h), ...]`.

        Returns:
            A dictionary containing the calculated metrics.
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
