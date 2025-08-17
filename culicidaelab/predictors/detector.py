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
from typing import Any, TypeAlias
from collections.abc import Sequence

import cv2
import numpy as np

# CHANGED: Replaced tqdm with fastprogress for consistency.
from fastprogress.fastprogress import progress_bar
from ultralytics import YOLO

from culicidaelab.core.base_predictor import BasePredictor, ImageInput

from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

logger = logging.getLogger(__name__)

DetectionPredictionType: TypeAlias = list[tuple[float, float, float, float, float]]
DetectionGroundTruthType: TypeAlias = list[tuple[float, float, float, float]]


class MosquitoDetector(
    BasePredictor[ImageInput, DetectionPredictionType, DetectionGroundTruthType],
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

    def __init__(self, settings: Settings, load_model: bool = False) -> None:
        """Initializes the MosquitoDetector."""

        weights_manager = ModelWeightsManager(
            settings=settings,
        )
        super().__init__(
            settings=settings,
            predictor_type="detector",
            weights_manager=weights_manager,
            load_model=load_model,
        )
        self.confidence_threshold: float = self.config.confidence or 0.5
        self.iou_threshold: float = self.config.params.get("iou_threshold", 0.45)
        self.max_detections: int = self.config.params.get("max_detections", 300)

    def predict(self, input_data: ImageInput, **kwargs: Any) -> DetectionPredictionType:
        """Detects mosquitos in a single image.

        Args:
            input_data (ImageInput): The input image as a NumPy array or other supported format.
            **kwargs (Any): Optional keyword arguments, including:
                confidence_threshold (float): Override the default confidence
                    threshold for this prediction.

        Returns:
            DetectionPredictionType: A list of detection tuples. Each tuple is
            (x1, y1, x2, y2, confidence). Returns an empty
            list if no mosquitos are found.

        Raises:
            RuntimeError: If the model fails to load or if prediction fails.
        """
        if not self.model_loaded or self._model is None:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Failed to load model")

        confidence_threshold = kwargs.get(
            "confidence_threshold",
            self.confidence_threshold,
        )

        try:
            input_data = np.array(self._load_and_validate_image(input_data))
            results = self._model(
                source=input_data,
                conf=confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}") from e

        detections: DetectionPredictionType = []
        if results:
            boxes = results[0].boxes
            for box in boxes:
                xyxy_tensor = box.xyxy[0]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()
                conf = float(box.conf[0])

                detections.append((x1, y1, x2, y2, conf))
        return detections

    def predict_batch(
        self,
        input_data_batch: Sequence[ImageInput],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[DetectionPredictionType]:
        """Detects mosquitos in a batch of images using YOLO's native batching.

        Args:
            input_data_batch (Sequence[np.ndarray]): A list of input images.
            show_progress (bool, optional): If True, a progress bar is shown.
                Defaults to True.
            **kwargs (Any): Additional arguments (not used).

        Returns:
            list[DetectionPredictionType]: A list where each item is the list
            of detections for the corresponding image in the input batch.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not input_data_batch:
            return []

        valid_images, valid_indices = self._prepare_batch_images(input_data_batch)

        if not valid_images:
            self._logger.warning("No valid images found in the batch to process.")
            return [[]] * len(input_data_batch)

        yolo_results = self._model(
            source=valid_images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            max_det=self.max_detections,
            stream=False,
            verbose=False,
        )

        all_predictions: list[DetectionPredictionType] = []
        # CHANGED: Refactored the loop to use progress_bar correctly.
        iterator = yolo_results
        if show_progress:
            iterator = progress_bar(
                yolo_results,
                total=len(input_data_batch),
            )

        for r in iterator:
            detections = []
            for box in r.boxes:
                xyxy_tensor = box.xyxy[0]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
            all_predictions.append(detections)

        return all_predictions

    def visualize(
        self,
        input_data: ImageInput,
        predictions: DetectionPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Draws predicted bounding boxes on an image.

        Args:
            input_data (ImageInput): The original image.
            predictions (DetectionPredictionType): The list of detections from `predict`.
            save_path (str | Path | None, optional): If provided, the output
                image is saved to this path. Defaults to None.

        Returns:
            np.ndarray: A new image array with bounding boxes and confidence
            scores drawn on it.
        """
        vis_img = np.array(self._load_and_validate_image(input_data))
        vis_config = self.config.visualization
        box_color = str_to_bgr(vis_config.box_color)
        text_color = str_to_bgr(vis_config.text_color)
        font_scale = vis_config.font_scale
        thickness = vis_config.box_thickness

        for x1, y1, x2, y2, conf in predictions:
            cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, thickness)
            text = f"{conf:.2f}"
            cv2.putText(
                vis_img,
                text,
                (int(x1), int(y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness,
            )

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return vis_img

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
        prediction: DetectionPredictionType,
        ground_truth: DetectionGroundTruthType,
    ) -> dict[str, float]:
        """Calculates detection metrics for a single image's predictions.

        This computes precision, recall, F1-score, Average Precision (AP),
        and mean IoU for a set of predicted boxes against ground truth boxes.

        Args:
            prediction (DetectionPredictionType): A list of predicted boxes with
                confidence scores: `[(x, y, w, h, conf), ...]`.
            ground_truth (DetectionGroundTruthType): A list of ground truth
                boxes: `[(x, y, w, h), ...]`.

        Returns:
            dict[str, float]: A dictionary containing the calculated metrics.
        """
        if not ground_truth and not prediction:
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
        if not prediction:  # False negatives exist
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "mean_iou": 0.0,
            }

        predictions_sorted = sorted(prediction, key=lambda x: x[4], reverse=True)
        tp = np.zeros(len(predictions_sorted))
        fp = np.zeros(len(predictions_sorted))
        gt_matched = [False] * len(ground_truth)
        all_ious_for_mean = []
        iou_threshold = self.iou_threshold

        for i, pred_box_with_conf in enumerate(predictions_sorted):
            pred_box = pred_box_with_conf[:4]
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

    def _load_model(self) -> None:
        """Loads the YOLO object detection model from the specified path.

        Raises:
            RuntimeError: If the model cannot be loaded from the path
                specified in the configuration.
        """
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self._model = YOLO(str(self.model_path), task="detect")

            if self._model and hasattr(self.config, "device") and self.config.device:
                device = str(self.config.device)
                logger.info(f"Moving model to device: {device}")
                self._model.to(device)

            logger.info("YOLO model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self._model = None
            raise RuntimeError(
                f"Could not load YOLO model from {self.model_path}.",
            ) from e
