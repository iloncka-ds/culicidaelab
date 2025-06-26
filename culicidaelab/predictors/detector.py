r"""
Module for mosquito detection in images using YOLO.
culicidaelab\predictors\detector.py
"""

from __future__ import annotations

from typing import Any, TypeAlias

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
from tqdm import tqdm

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.utils import str_to_bgr
from .model_weights_manager import ModelWeightsManager

logger = logging.getLogger(__name__)

DetectionPredictionType: TypeAlias = list[tuple[float, float, float, float, float]]
DetectionGroundTruthType: TypeAlias = list[tuple[float, float, float, float]]


class MosquitoDetector(BasePredictor[DetectionPredictionType, DetectionGroundTruthType]):
    """
    Class for detecting mosquitos in images using YOLO.
    This class correctly implements the BasePredictor interface.
    """

    def __init__(
        self,
        settings: Settings,
        weights_manager: ModelWeightsManager,
        load_model: bool = False,
    ) -> None:
        """
        Initialize the mosquito detector.
        Args:
            settings: The main Settings object for the library.
            load_model: Whether to load the model immediately.

        """
        provider_service = ProviderService(settings)
        weights_manager = ModelWeightsManager(
            settings=settings,
            provider_service=provider_service,
        )
        super().__init__(
            settings=settings,
            predictor_type="detector",
            weights_manager=weights_manager,  # Use the injected dependency
            load_model=load_model,
        )
        self.confidence_threshold = self.config.confidence or 0.5

        self.iou_threshold = self.config.params.get("iou_threshold", 0.45)
        self.max_detections = self.config.params.get("max_detections", 300)

    def _load_model(self) -> None:
        """Load the YOLO model."""
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self._model = YOLO(str(self.model_path), task="detect")

            # Move model to device if specified in config
            if self._model and hasattr(self.config, "device") and self.config.device:
                device = str(self.config.device)
                logger.info(f"Moving model to device: {device}")
                self._model.to(device)

            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self._model = None
            raise RuntimeError(f"Could not load YOLO model from {self.model_path}.") from e

    def predict(self, input_data: np.ndarray, **kwargs: Any) -> DetectionPredictionType:
        """
        Detect mosquitos in a single image.

        Args:
            input_data: The input image as a NumPy array.
            **kwargs: Additional arguments including:
                confidence_threshold: Optional confidence threshold override

        Returns:
            A list of detections, where each detection is a tuple of
            (center_x, center_y, width, height, confidence).
        """
        if not self.model_loaded or self._model is None:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Failed to load model")

        confidence_threshold = kwargs.get("confidence_threshold", self.confidence_threshold)

        # predict() from ultralytics can take a single image
        try:
            # Get config values with defaults
            iou_threshold = getattr(self.config, "params", {}).get("iou_threshold", 0.45)
            max_detections = getattr(self.config, "params", {}).get("max_detections", 300)

            results = self._model(
                source=input_data,
                conf=confidence_threshold,
                iou=iou_threshold,
                max_det=max_detections,
                verbose=False,  # Reduce console noise
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}") from e

        detections: DetectionPredictionType = []
        # Results list will have one element for a single image
        if results:
            boxes = results[0].boxes
            for box in boxes:
                xyxy_tensor = box.xyxy[0]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()
                conf = float(box.conf[0])
                w, h = x2 - x1, y2 - y1
                center_x, center_y = x1 + w / 2, y1 + h / 2
                detections.append((center_x, center_y, w, h, conf))
        return detections

    def predict_batch(
        self,
        input_data_batch: list[np.ndarray],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[DetectionPredictionType]:
        """
        Make predictions on a batch of inputs using YOLO's native batching for performance.

        Args:
            input_data_batch: List of input data to make predictions on.
            show_progress: Whether to show a progress bar (Note: internal batching may affect bar).

        Returns:
            A list of predictions, where each item is the prediction list for an image.
        """
        if not self.model_loaded:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        all_predictions: list[DetectionPredictionType] = []

        # Use tqdm for progress tracking if requested
        iterator = range(0, len(input_data_batch))
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting detection batch")

        # The model handles the batching internally.
        # verbose=False prevents ultralytics from printing its own status lines.
        yolo_results = self._model(
            source=input_data_batch,
            conf=self.confidence_threshold,
            iou=self.config.params["iou_threshold"],
            max_det=self.config.params["max_detections"],
            stream=False,  # Process all at once
            verbose=False,
        )

        for i, r in enumerate(yolo_results):
            detections = []
            for box in r.boxes:
                xyxy_tensor = box.xyxy[0]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()
                w, h = x2 - x1, y2 - y1
                center_x, center_y = x1 + w / 2, y1 + h / 2
                conf = float(box.conf[0])
                detections.append((center_x, center_y, w, h, conf))
            all_predictions.append(detections)
            if show_progress:
                iterator.update(1)
        if show_progress:
            iterator.close()

        return all_predictions

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: DetectionPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Visualizes detections on an image by drawing bounding boxes."""
        vis_img = input_data.copy()
        vis_config = self.config.visualization
        box_color = str_to_bgr(vis_config.box_color)
        text_color = str_to_bgr(vis_config.text_color)
        font_scale = vis_config.font_scale
        thickness = vis_config.thickness

        for x, y, w, h, conf in predictions:
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color, thickness)
            text = f"{conf:.2f}"
            cv2.putText(
                vis_img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
            )

        if save_path:
            # OpenCV expects BGR format for saving
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return vis_img

    def _evaluate_from_prediction(
        self,
        prediction: DetectionPredictionType,
        ground_truth: DetectionGroundTruthType,
    ) -> dict[str, float]:
        """
        The core metric calculation logic for a single item.
        Calculates precision, recall, F1, AP, and mean IoU for detections.

        Args:
            prediction: A list of predicted boxes with confidence: [(x, y, w, h, conf), ...].
            ground_truth: A list of ground truth boxes: [(x, y, w, h), ...].

        Returns:
            A dictionary containing evaluation metrics for a single image.
        """
        # Type checking for clarity
        predictions_with_conf: DetectionPredictionType = prediction
        gt_boxes: DetectionGroundTruthType = ground_truth

        if not gt_boxes and not predictions_with_conf:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "ap": 1.0, "mean_iou": 0.0}
        if not gt_boxes:  # All predictions are false positives
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "mean_iou": 0.0}
        if not predictions_with_conf:  # All ground truths are false negatives
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "mean_iou": 0.0}

        # Sort predictions by confidence score in descending order
        predictions_sorted = sorted(predictions_with_conf, key=lambda x: x[4], reverse=True)

        tp = np.zeros(len(predictions_sorted))
        fp = np.zeros(len(predictions_sorted))
        gt_matched = [False] * len(gt_boxes)
        all_ious_for_mean = []
        iou_threshold = self.config.params["iou_threshold"] or 0.5

        for i, pred_box_with_conf in enumerate(predictions_sorted):
            pred_box = pred_box_with_conf[:4]
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
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
                else:  # Matched a GT that was already claimed by a higher-conf prediction
                    fp[i] = 1
            else:
                fp[i] = 1

        mean_iou_val = float(np.mean(all_ious_for_mean)) if all_ious_for_mean else 0.0

        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recall_curve = tp_cumsum / len(gt_boxes)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

        # Average Precision (AP) calculation using 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            precisions_at_recall_t = precision_curve[recall_curve >= t]
            ap += np.max(precisions_at_recall_t) if len(precisions_at_recall_t) > 0 else 0.0
        ap /= 11.0

        # Final metrics based on all predictions
        final_precision = precision_curve[-1] if len(precision_curve) > 0 else 0.0
        final_recall = recall_curve[-1] if len(recall_curve) > 0 else 0.0
        f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-9)

        return {
            "precision": float(final_precision),
            "recall": float(final_recall),
            "f1": float(f1),
            "ap": float(ap),
            "mean_iou": mean_iou_val,
        }

    def _calculate_iou(self, box1_xywh: tuple, box2_xywh: tuple) -> float:
        """Calculates Intersection over Union (IoU) for two boxes in (cx, cy, w, h) format."""
        # Convert (center_x, center_y, w, h) to (x1, y1, x2, y2)
        b1_x1, b1_y1 = box1_xywh[0] - box1_xywh[2] / 2, box1_xywh[1] - box1_xywh[3] / 2
        b1_x2, b1_y2 = box1_xywh[0] + box1_xywh[2] / 2, box1_xywh[1] + box1_xywh[3] / 2

        b2_x1, b2_y1 = box2_xywh[0] - box2_xywh[2] / 2, box2_xywh[1] - box2_xywh[3] / 2
        b2_x2, b2_y2 = box2_xywh[0] + box2_xywh[2] / 2, box2_xywh[1] + box2_xywh[3] / 2

        # Calculate intersection area
        inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
        inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
        intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Calculate union area
        area1, area2 = box1_xywh[2] * box1_xywh[3], box2_xywh[2] * box2_xywh[3]
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0
