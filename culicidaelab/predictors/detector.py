"""
Module for mosquito detection in images using YOLO.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Any
from pathlib import Path
import logging

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.config_manager import ConfigManager


logger = logging.getLogger(__name__)


class MosquitoDetector(BasePredictor):
    """Class for detecting mosquitos in images using YOLO."""

    def __init__(self, model_path: str | Path, config_manager: ConfigManager) -> None:
        """
        Initialize the mosquito detector.

        Args:
            model_path: Path to pre-trained YOLO model weights
            config_manager: Configuration manager instance
        """
        super().__init__(model_path, config_manager=config_manager)

        self.load_config(config_path=None)

        if self.config is None or not hasattr(self.config, "model"):
            raise ValueError(
                "Detector configuration is missing or does not have a 'model' section. "
                "Ensure ConfigManager is correctly set up and 'detector' config is available.",
            )

        self.confidence_threshold = self.config.model.confidence_threshold
        self._model: YOLO | None = None

    def _load_model(self) -> None:
        """Load the YOLO model."""
        logger.info(f"Loading YOLO model from: {self.model_path}")
        self._model = YOLO(str(self.model_path))
        if hasattr(self.config.model, "device") and self.config.model.device:
            logger.info(f"Moving model to device: {self.config.model.device}")
            self._model.to(self.config.model.device)
        self.model_loaded = True
        logger.info("YOLO model loaded successfully.")

    def predict(self, input_data: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        Detect mosquitos in an image.
        """
        if not self.model_loaded:
            self.load_model()

        if self._model is None:
            logger.error("Model was not loaded before predict was called.")
            raise RuntimeError("Model not loaded. Call load_model() first.")

        results = self._model(
            input_data,
            conf=self.confidence_threshold,
            iou=self.config.model.iou_threshold,
            max_det=self.config.model.max_detections,
        )

        detections: list[tuple[float, float, float, float, float]] = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy_tensor = box.xyxy[0]
                x1, y1, x2, y2 = xyxy_tensor.cpu().numpy()

                w = x2 - x1
                h = y2 - y1
                center_x = x1 + w / 2
                center_y = y1 + h / 2

                conf = float(box.conf[0])
                detections.append((center_x, center_y, w, h, conf))
        return detections

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: list[tuple[float, float, float, float, float]],
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        vis_img = input_data.copy()
        for x, y, w, h, conf in predictions:
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            box_color_bgr = self.config.visualization.box_color
            if isinstance(box_color_bgr, str):
                hex_color = box_color_bgr.lstrip("#")
                box_color_bgr = tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

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
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        return vis_img

    def evaluate(
        self,
        input_data: np.ndarray,
        ground_truth: list[tuple[float, float, float, float]],
    ) -> dict[str, float]:
        """
        Evaluate detection predictions against ground truth boxes.
        Note: This is a simplified evaluation. For rigorous mAP, use `evaluate_batch` or dedicated libraries.
        """
        predictions_with_conf = self.predict(input_data)

        if not ground_truth and not predictions_with_conf:
            return {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "ap": 1.0,
                "mean_iou": 0.0,
            }
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

        if not all_ious_for_mean:
            mean_iou_val = 0.0
        else:
            mean_iou_val = float(np.mean(all_ious_for_mean)) if all_ious_for_mean else 0.0

        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recall_curve = tp_cumsum / len(ground_truth)
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
            "precision": float(final_precision),
            "recall": float(final_recall),
            "f1": float(f1),
            "ap": float(ap),
            "mean_iou": mean_iou_val,
        }

    def evaluate_batch(
        self,
        input_data_batch: list[np.ndarray],
        ground_truth_batch: list[Any],
        num_workers: int = 4,
        batch_size: int = 32,
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

        per_image_metrics_list = []

        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        def evaluate_single_with_preds(idx: int) -> dict[str, float]:
            def _calculate_metrics_from_predictions(
                predictions_with_conf: list[tuple[float, float, float, float, float]],
                ground_truth: list[tuple[float, float, float, float]],
            ) -> dict[str, float]:
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

            return _calculate_metrics_from_predictions(all_predictions_batched[idx], ground_truth_batch[idx])

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

        return super()._aggregate_metrics(per_image_metrics_list)

    def _aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics from multiple evaluations (e.g., batches or images).
        Calculates the mean of each metric.
        """
        if not metrics_list:
            return {}

        valid_metrics_list = [m for m in metrics_list if isinstance(m, dict) and m]
        if not valid_metrics_list:
            return {}

        aggregated_metrics: dict[str, float] = {}
        sample_keys = valid_metrics_list[0].keys()

        for key in sample_keys:
            values = [
                metrics.get(key, 0.0) for metrics in valid_metrics_list if isinstance(metrics.get(key), (int, float))
            ]
            if values:
                aggregated_metrics[key] = float(np.mean(values))
            else:
                aggregated_metrics[key] = 0.0
        return aggregated_metrics

    def _calculate_iou(self, box1_xywh: list[float], box2_xywh: list[float]) -> float:
        b1_x1, b1_y1 = box1_xywh[0] - box1_xywh[2] / 2, box1_xywh[1] - box1_xywh[3] / 2
        b1_x2, b1_y2 = box1_xywh[0] + box1_xywh[2] / 2, box1_xywh[1] + box1_xywh[3] / 2

        b2_x1, b2_y1 = box2_xywh[0] - box2_xywh[2] / 2, box2_xywh[1] - box2_xywh[3] / 2
        b2_x2, b2_y2 = box2_xywh[0] + box2_xywh[2] / 2, box2_xywh[1] + box2_xywh[3] / 2

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
