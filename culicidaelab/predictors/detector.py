"""
Module for mosquito detection in images using YOLO.
"""

from __future__ import annotations

import cv2  # Keep for visualization
import numpy as np
from ultralytics import YOLO  # Ensure this is imported
from typing import Any  # Use precise List, Tuple, Dict
from pathlib import Path
import logging  # Import logging

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.config_manager import ConfigManager

# ThreadPoolExecutor and tqdm are used by BasePredictor, not directly here unless overriding batch methods extensively.

logger = logging.getLogger(__name__)  # Setup logger


class MosquitoDetector(BasePredictor):
    """Class for detecting mosquitos in images using YOLO."""

    def __init__(self, model_path: str | Path, config_manager: ConfigManager) -> None:
        """
        Initialize the mosquito detector.

        Args:
            model_path: Path to pre-trained YOLO model weights
            config_manager: Configuration manager instance
        """
        # It's good practice for components to declare their config needs or key.
        # For now, we'll stick to the original direct config access pattern.
        super().__init__(model_path, config_manager=config_manager)  # Pass config_manager correctly

        # The original MosquitoDetector directly gets the whole config and assumes
        # the necessary 'model' structure is at its top level.
        # This happens because BasePredictor.__init__ calls super().__init__(config_manager)
        # which calls ConfigurableComponent.__init__(config_manager, component_config_key=None (default))
        # Then, self.load_config() in ConfigurableComponent (if called, or if self._config is accessed via property)
        # would set self._config.
        # MosquitoDetector's original: self._config = self.config_manager.get_config()
        # This sets _config directly. Let's make it use the ConfigurableComponent pattern.

        # OPTION 1 (Minimal change to existing logic, but less aligned with ConfigurableComponent intended use):
        # self._config = self.config_manager.get_config("detector") # Assuming get_config can take a key
        # if not self._config: # If get_config(key) can return None
        #     self._config = self.config_manager.get_config() # Fallback to whole config

        # OPTION 2 (Align with ConfigurableComponent and BasePredictor structure):
        # Define a key for this predictor's config section
        # This key should match how your configs are structured, e.g., conf/predictors/detector.yaml
        # or a section in the main config.yaml:
        # predictors:
        #   detector:
        #     model: ...
        #     visualization: ...
        # For the test setup, "detector" seems like a reasonable key.
        # We need to ensure BasePredictor's __init__ accepts predictor_config_key
        # For now, let's assume the config is loaded and is directly what the detector needs.
        # The test fixture currently makes `config_manager.get_config()` return the component's config.

        self.load_config(config_path=None)  # Load the entire config into self._config via ConfigurableComponent
        # Or, if a key like "detector" is used: self.load_config("detector")
        # For now, using config_path=None. Test fixture will mock get_config().

        if self.config is None or not hasattr(self.config, "model"):
            raise ValueError(
                "Detector configuration is missing or does not have a 'model' section. "
                "Ensure ConfigManager is correctly set up and 'detector' config is available.",
            )

        self.confidence_threshold = self.config.model.confidence_threshold
        self._model: YOLO | None = None  # Initialize model attribute

    def _load_model(self) -> None:
        """Load the YOLO model."""
        logger.info(f"Loading YOLO model from: {self.model_path}")
        self._model = YOLO(str(self.model_path))  # Assign to self._model
        if hasattr(self.config.model, "device") and self.config.model.device:  # Check for device attr
            logger.info(f"Moving model to device: {self.config.model.device}")
            self._model.to(self.config.model.device)
        self.model_loaded = True  # CRITICAL: Set model_loaded to True
        logger.info("YOLO model loaded successfully.")

    def predict(self, input_data: np.ndarray) -> list[tuple[float, float, float, float, float]]:  # Precise types
        """
        Detect mosquitos in an image.
        """
        if not self.model_loaded:
            self.load_model()  # Calls BasePredictor.load_model -> self._load_model

        if self._model is None:
            logger.error("Model was not loaded before predict was called.")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self._model(  # Use self._model
            input_data,
            conf=self.confidence_threshold,  # Already set from self.config.model.confidence_threshold
            iou=self.config.model.iou_threshold,
            max_det=self.config.model.max_detections,
        )

        detections: list[tuple[float, float, float, float, float]] = []
        for r in results:  # result objects from ultralytics
            boxes = r.boxes  # Boxes object
            for box in boxes:  # Iterating through individual BoundingBox objects
                # Get box coordinates in xyxy format
                xyxy_tensor = box.xyxy[0]  # Should be a tensor [x1, y1, x2, y2]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()

                w = x2 - x1
                h = y2 - y1
                # Center x, y
                center_x = x1 + w / 2
                center_y = y1 + h / 2

                conf = float(box.conf[0])  # Confidence score
                detections.append((center_x, center_y, w, h, conf))
        return detections

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: list[tuple[float, float, float, float, float]],  # Precise type
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        vis_img = input_data.copy()
        for x, y, w, h, conf in predictions:
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            box_color_bgr = self.config.visualization.box_color
            if isinstance(box_color_bgr, str):  # If color is hex string from config
                # Basic hex to BGR conversion (assumes #RRGGBB)
                hex_color = box_color_bgr.lstrip("#")
                box_color_bgr = tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))  # BGR

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), box_color_bgr, self.config.visualization.box_thickness)
            text = f"{conf:.2f}"
            cv2.putText(
                vis_img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.visualization.font_scale,
                self.config.visualization.text_color,
                self.config.visualization.text_thickness,
            )
        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))  # Assuming vis_img is RGB
        return vis_img

    def evaluate(
        self,
        input_data: np.ndarray,
        ground_truth: list[tuple[float, float, float, float]],  # GT format: center_x, center_y, w, h
    ) -> dict[str, float]:  # Precise type
        """
        Evaluate detection predictions against ground truth boxes.
        Note: This is a simplified evaluation. For rigorous mAP, use `evaluate_batch` or dedicated libraries.
        """
        # The current implementation of self.predict calls self.load_model if not loaded.
        predictions_with_conf = self.predict(input_data)  # List of (x, y, w, h, conf)

        # This evaluation is per-image and might not align with standard mAP calculations
        # which typically operate over a whole dataset.
        # For simplicity, this calculates IoU-based precision/recall for this single image.

        # Let's adapt from the evaluate_batch logic for a single image's AP.
        if not ground_truth and not predictions_with_conf:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "ap": 1.0,
                "mean_iou": 0.0,
            }  # Or 0.0 if preferred for empty
        if not ground_truth:  # No ground truth, all predictions are false positives if any
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "mean_iou": 0.0}
        if not predictions_with_conf:  # No predictions, all ground truths are false negatives
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "mean_iou": 0.0}

        # Sort predictions by confidence
        predictions_sorted = sorted(predictions_with_conf, key=lambda x: x[4], reverse=True)

        tp = np.zeros(len(predictions_sorted))
        fp = np.zeros(len(predictions_sorted))
        gt_matched = [False] * len(ground_truth)
        all_ious_for_mean = []

        iou_threshold = self.config.evaluation.get("iou_threshold", 0.5)  # Get from config, with default

        for i, pred_box_with_conf in enumerate(predictions_sorted):
            pred_box = pred_box_with_conf[:4]  # x, y, w, h
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt_box in enumerate(ground_truth):
                if not gt_matched[j]:
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

            if best_gt_idx != -1:  # Store IoU if there was any overlap with any GT box
                all_ious_for_mean.append(best_iou)

            if best_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp[i] = 1
                    gt_matched[best_gt_idx] = True
                else:  # Matched a GT box that was already matched by a higher confidence prediction
                    fp[i] = 1
            else:
                fp[i] = 1

        if not all_ious_for_mean:  # if no predictions or no overlaps
            mean_iou_val = 0.0
        else:
            mean_iou_val = float(np.mean(all_ious_for_mean)) if all_ious_for_mean else 0.0

        # Compute precision and recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recall_curve = tp_cumsum / len(ground_truth)
        precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)  # Avoid division by zero

        # AP calculation (Area under PR curve)
        ap = 0.0
        # Using 11-point interpolation (common for VOC PASCAL)
        for t in np.linspace(0, 1, 11):
            precisions_at_recall_t = precision_curve[recall_curve >= t]
            ap += np.max(precisions_at_recall_t) if len(precisions_at_recall_t) > 0 else 0.0
        ap /= 11.0

        final_precision = precision_curve[-1] if len(precision_curve) > 0 else 0.0
        final_recall = recall_curve[-1] if len(recall_curve) > 0 else 0.0
        f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-9)

        return {
            "precision": float(final_precision),
            "recall": float(final_recall),
            "f1": float(f1),
            "ap": float(ap),  # Average Precision for this image
            "mean_iou": mean_iou_val,
        }

    def evaluate_batch(
        self,
        input_data_batch: list[np.ndarray],  # Precise types
        ground_truth_batch: list[Any],  # Or List[List[Tuple[...]]]
        num_workers: int = 4,  # Passed to BasePredictor's implementation
        batch_size: int = 32,  # Used by BasePredictor's implementation
    ) -> dict[str, float]:
        """
        Evaluate model on a batch of inputs using parallel processing.
        This implementation overrides BasePredictor.evaluate_batch to use YOLO's
        own batch prediction capabilities for efficiency, then evaluates metrics.
        """
        if len(input_data_batch) != len(ground_truth_batch):
            raise ValueError("Number of inputs must match number of ground truth annotations")
        if not input_data_batch:
            logger.warning("Input batch for evaluation is empty.")
            return {}

        if not self.model_loaded:
            self.load_model()
        if self._model is None:
            raise RuntimeError("Model not loaded for batch evaluation.")

        # 1. Get all predictions using YOLO's batching
        all_predictions_batched: list[list[tuple[float, float, float, float, float]]] = []
        # Determine YOLO's internal batch size for prediction if different from metric calculation batch_size
        yolo_predict_batch_size = self.config.model.get("predict_batch_size", batch_size)

        for i in range(0, len(input_data_batch), yolo_predict_batch_size):
            current_input_batch = input_data_batch[i : i + yolo_predict_batch_size]
            yolo_results = self._model(
                current_input_batch,
                conf=self.confidence_threshold,
                iou=self.config.model.iou_threshold,
                max_det=self.config.model.max_detections,
            )
            for r_idx, r in enumerate(yolo_results):
                detections = []
                for box in r.boxes:
                    xyxy_tensor = box.xyxy[0]
                    x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    center_x = x1 + w / 2
                    center_y = y1 + h / 2
                    conf = float(box.conf[0])
                    detections.append((center_x, center_y, w, h, conf))
                all_predictions_batched.append(detections)

        # 2. Evaluate each image's predictions against its ground truth
        # This part can be parallelized as in BasePredictor.evaluate_batch
        # We'll use a helper or call self.evaluate for each.

        per_image_metrics_list = []

        # Re-using the parallel execution structure from BasePredictor.evaluate_batch
        # but calling self.evaluate(image, gt) with pre-computed predictions.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        def evaluate_single_with_preds(idx: int) -> dict[str, float]:
            # self.evaluate expects an image to run predict on, but we already have predictions.
            # So, we need a modified evaluation function or adapt self.evaluate.
            # For now, let's assume self.evaluate can be used if it's okay to re-predict (less efficient)
            # OR, we create a temporary method here that takes predictions directly.
            # For this fix, let's use self.evaluate.
            # This means `predict` is called again for each image, which is not ideal if we already batched.
            # A better way: have a separate _calculate_metrics(predictions, ground_truth) method.

            # Let's create a temporary local function for metric calculation from preds
            def _calculate_metrics_from_predictions(
                predictions_with_conf: list[tuple[float, float, float, float, float]],
                ground_truth: list[tuple[float, float, float, float]],
            ) -> dict[str, float]:
                # This is essentially a copy of the logic from self.evaluate,
                # but takes predictions as input.
                if not ground_truth and not predictions_with_conf:
                    return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "ap": 1.0, "mean_iou": 0.0}
                if not ground_truth:
                    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "mean_iou": 0.0}
                if not predictions_with_conf:
                    return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0, "mean_iou": 0.0}

                predictions_sorted = sorted(predictions_with_conf, key=lambda x: x[4], reverse=True)
                tp = np.zeros(len(predictions_sorted))
                fp = np.zeros(len(predictions_sorted))
                gt_matched = [False] * len(ground_truth)
                all_ious_for_mean = []
                iou_threshold = self.config.evaluation.get("iou_threshold", 0.5)

                for i, pred_box_with_conf in enumerate(predictions_sorted):
                    pred_box = pred_box_with_conf[:4]
                    best_iou = 0.0
                    best_gt_idx = -1
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
                        else:
                            fp[i] = 1
                    else:
                        fp[i] = 1

                mean_iou_val = float(np.mean(all_ious_for_mean)) if all_ious_for_mean else 0.0
                fp_cumsum = np.cumsum(fp)
                tp_cumsum = np.cumsum(tp)
                recall_curve = tp_cumsum / (len(ground_truth) + 1e-9)
                precision_curve = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
                ap = 0.0
                for t in np.linspace(0, 1, 11):
                    precisions_at_recall_t = precision_curve[recall_curve >= t]
                    ap += np.max(precisions_at_recall_t) if len(precisions_at_recall_t) > 0 else 0.0
                ap /= 11.0
                final_precision = precision_curve[-1] if len(precision_curve) > 0 else 0.0
                final_recall = recall_curve[-1] if len(recall_curve) > 0 else 0.0
                f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall + 1e-9)
                return {
                    "precision": final_precision,
                    "recall": final_recall,
                    "f1": f1,
                    "ap": ap,
                    "mean_iou": mean_iou_val,
                }

            # End of _calculate_metrics_from_predictions helper

            # Calculate metrics for the current image using its pre-computed predictions
            # The `idx` here corresponds to the original full batch, not the ThreadPoolExecutor's batching.
            # We need to map `idx` (from range(len(input_data_batch))) to the ThreadPool batch.
            # The original BasePredictor.evaluate_batch function `process_batch` receives `batch_idx`.
            # This override aims to replace that `process_batch`.

            # Let's iterate through images and their predictions for evaluation.
            # The parallelization will be over these individual evaluations.
            # The `idx` here is the index into `input_data_batch`.
            return _calculate_metrics_from_predictions(all_predictions_batched[idx], ground_truth_batch[idx])

        # Determine number of tasks for parallel execution (one per image)
        num_eval_tasks = len(input_data_batch)
        image_indices = range(num_eval_tasks)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(evaluate_single_with_preds, idx) for idx in image_indices]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating detection batches"):
                try:
                    per_image_metrics_list.append(future.result())
                except Exception as e:
                    logger.error(f"Error evaluating image in batch: {e}", exc_info=True)

        if not per_image_metrics_list:
            logger.warning("No metrics were calculated during batch evaluation.")
            return {}

        # Aggregate metrics from all images using BasePredictor's aggregation
        return super()._aggregate_metrics(per_image_metrics_list)

    # _calculate_iou method seems fine.
    # The _aggregate_metrics in MosquitoDetector was different from BasePredictor.
    # If BasePredictor._aggregate_metrics (mean + std) is desired, remove the override.
    # If only mean is desired (as in the original MosquitoDetector._aggregate_metrics), keep it.
    # For AP/mAP, typically you average the AP scores, so mean is appropriate.
    # Let's assume the override is intentional for mAP style aggregation.
    def _aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics from multiple evaluations (e.g., batches or images).
        Calculates the mean of each metric.
        """
        if not metrics_list:
            return {}

        # Ensure all metric dicts are valid before processing
        valid_metrics_list = [m for m in metrics_list if isinstance(m, dict) and m]
        if not valid_metrics_list:
            return {}

        aggregated_metrics: dict[str, float] = {}
        # Get all unique metric keys from the first valid entry
        # (assuming all dicts have similar keys, or handle missing keys)
        sample_keys = valid_metrics_list[0].keys()

        for key in sample_keys:
            values = [
                metrics.get(key, 0.0) for metrics in valid_metrics_list if isinstance(metrics.get(key), (int, float))
            ]
            if values:
                aggregated_metrics[key] = float(np.mean(values))
            else:  # Handle case where a key might not be present or has non-numeric values across all items
                aggregated_metrics[key] = 0.0
        return aggregated_metrics

    def _calculate_iou(self, box1_xywh: list[float], box2_xywh: list[float]) -> float:
        # Convert center_x, center_y, w, h to x1, y1, x2, y2
        b1_x1, b1_y1 = box1_xywh[0] - box1_xywh[2] / 2, box1_xywh[1] - box1_xywh[3] / 2
        b1_x2, b1_y2 = box1_xywh[0] + box1_xywh[2] / 2, box1_xywh[1] + box1_xywh[3] / 2

        b2_x1, b2_y1 = box2_xywh[0] - box2_xywh[2] / 2, box2_xywh[1] - box2_xywh[3] / 2
        b2_x2, b2_y2 = box2_xywh[0] + box2_xywh[2] / 2, box2_xywh[1] + box2_xywh[3] / 2

        # Intersection
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area1 = box1_xywh[2] * box1_xywh[3]
        area2 = box2_xywh[2] * box2_xywh[3]
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0
