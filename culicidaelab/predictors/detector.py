"""
Module for mosquito detection in images using YOLO.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Any
import torch
from pathlib import Path
from culicidaelab.core._base_predictor import BasePredictor
from culicidaelab.core.config_manager import ConfigManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class MosquitoDetector(BasePredictor):
    """Class for detecting mosquitos in images using YOLO."""

    def __init__(self, model_path: str | Path, config_manager: ConfigManager) -> None:
        """
        Initialize the mosquito detector.

        Args:
            model_path: Path to pre-trained YOLO model weights
            config_manager: Configuration manager instance
        """
        super().__init__(model_path, config_manager)
        self.config = self.config_manager.get_config()
        self.confidence_threshold = self.config.model.confidence_threshold

    def _load_model(self) -> None:
        """Load the YOLO model."""
        self.model = YOLO(str(self.model_path))
        if self.config.model.device:
            self.model.to(self.config.model.device)

    def predict(self, input_data: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        Detect mosquitos in an image.

        Args:
            input_data: Input image as numpy array

        Returns:
            List of tuples (x, y, width, height, confidence)
        """
        if not self.model_loaded:
            self.load_model()

        # Run inference
        results = self.model(
            input_data,
            conf=self.confidence_threshold,
            iou=self.config.model.iou_threshold,
            max_det=self.config.model.max_detections,
        )

        # Process predictions
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Convert to center format
                w = x2 - x1
                h = y2 - y1
                x = x1 + w / 2
                y = y1 + h / 2

                # Get confidence
                conf = float(box.conf[0])

                detections.append((x, y, w, h, conf))

        return detections

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: list[tuple[float, float, float, float, float]],
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Visualize detections on the image.

        Args:
            input_data: Original image
            predictions: List of (x, y, width, height, confidence) tuples
            save_path: Optional path to save visualization

        Returns:
            np.ndarray: Image with visualized detections
        """
        # Create visualization
        vis_img = input_data.copy()

        # Draw each detection
        for x, y, w, h, conf in predictions:
            # Convert center format to corners
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            # Draw rectangle
            cv2.rectangle(
                vis_img,
                (x1, y1),
                (x2, y2),
                self.config.visualization.box_color,
                self.config.visualization.box_thickness,
            )

            # Add confidence text
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

        Args:
            input_data: Input image
            ground_truth: List of ground truth boxes (x, y, width, height)

        Returns:
            Dictionary containing mAP and other metrics
        """
        predictions = self.predict(input_data)

        # Calculate metrics
        metrics = {}

        # Calculate IoU for each prediction-ground truth pair
        ious = []
        for gt_box in ground_truth:
            gt_x, gt_y, gt_w, gt_h = gt_box

            # Convert ground truth to corners
            gt_x1 = gt_x - gt_w / 2
            gt_y1 = gt_y - gt_h / 2
            gt_x2 = gt_x + gt_w / 2
            gt_y2 = gt_y + gt_h / 2

            box_ious = []
            for pred_box in predictions:
                pred_x, pred_y, pred_w, pred_h, _ = pred_box

                # Convert prediction to corners
                pred_x1 = pred_x - pred_w / 2
                pred_y1 = pred_y - pred_h / 2
                pred_x2 = pred_x + pred_w / 2
                pred_y2 = pred_y + pred_h / 2

                # Calculate intersection
                inter_x1 = max(gt_x1, pred_x1)
                inter_y1 = max(gt_y1, pred_y1)
                inter_x2 = min(gt_x2, pred_x2)
                inter_y2 = min(gt_y2, pred_y2)

                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    gt_area = gt_w * gt_h
                    pred_area = pred_w * pred_h
                    union_area = gt_area + pred_area - inter_area
                    iou = inter_area / union_area
                else:
                    iou = 0.0

                box_ious.append(iou)

            ious.append(max(box_ious) if box_ious else 0.0)

        # Calculate metrics
        metrics["mean_iou"] = float(np.mean(ious)) if ious else 0.0
        metrics["recall"] = (
            float(len([iou for iou in ious if iou > self.config.evaluation.iou_threshold])) / len(ground_truth)
            if ground_truth
            else 0.0
        )
        metrics["precision"] = (
            float(len([iou for iou in ious if iou > self.config.evaluation.iou_threshold])) / len(predictions)
            if predictions
            else 0.0
        )

        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
        else:
            metrics["f1"] = 0.0

        return metrics

    def evaluate_batch(
        self,
        input_data_batch: list[np.ndarray],
        ground_truth_batch: list[Any],
        num_workers: int = 4,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """
        Optimized batch evaluation for detector that processes predictions in batches.

        Args:
            input_data_batch: List of input images
            ground_truth_batch: List of ground truth annotations
            num_workers: Number of parallel workers
            batch_size: Size of batches to process at once

        Returns:
            Dictionary containing aggregated evaluation metrics
        """
        if len(input_data_batch) != len(ground_truth_batch):
            raise ValueError("Number of inputs must match number of ground truth annotations")

        if len(input_data_batch) == 0:
            raise ValueError("Input batch cannot be empty")

        # Get predictions for entire batch at once using YOLO's built-in batching
        all_predictions = []
        for i in range(0, len(input_data_batch), batch_size):
            batch = input_data_batch[i : i + batch_size]
            results = self.model(batch, conf=self.confidence_threshold)

            # Process each result in the batch
            for r in results:
                boxes = r.boxes
                detections = []
                for box in boxes:
                    # Get box coordinates and confidence
                    xyxy = box.xyxy  # Shape: (1, 4)
                    if isinstance(xyxy, torch.Tensor):
                        x1, y1, x2, y2 = xyxy[0].cpu().numpy()
                    else:
                        x1, y1, x2, y2 = xyxy[0]
                    conf = float(box.conf[0])

                    # Convert to center format
                    w = x2 - x1
                    h = y2 - y1
                    x = x1 + w / 2
                    y = y1 + h / 2

                    detections.append((x, y, w, h, conf))
                all_predictions.append(detections)

        # Process evaluation in parallel
        def process_batch(batch_idx: int) -> dict[str, float]:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_data_batch))

            batch_metrics = []
            for i in range(start_idx, end_idx):
                pred_boxes = all_predictions[i]
                gt_boxes = ground_truth_batch[i]

                # Handle empty cases
                if not gt_boxes and not pred_boxes:
                    metrics = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "ap": 0.0,
                        "map": 0.0,
                    }
                    batch_metrics.append(metrics)
                    continue

                if not gt_boxes or not pred_boxes:
                    metrics = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "ap": 0.0,
                        "map": 0.0,
                    }
                    batch_metrics.append(metrics)
                    continue

                # Sort predictions by confidence
                pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

                # Initialize metrics
                tp = np.zeros(len(pred_boxes), dtype=np.float32)
                fp = np.zeros(len(pred_boxes), dtype=np.float32)
                used_gt = set()

                # Calculate IoU for each prediction
                for j, pred in enumerate(pred_boxes):
                    max_iou = 0.0
                    max_idx = -1

                    for k, gt in enumerate(gt_boxes):
                        if k in used_gt:
                            continue

                        iou = self._calculate_iou(pred[:4], gt)
                        if iou > max_iou:
                            max_iou = iou
                            max_idx = k

                    if max_iou >= 0.5:
                        tp[j] = 1.0
                        used_gt.add(max_idx)
                    else:
                        fp[j] = 1.0

                # Calculate cumulative metrics
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recall = cum_tp / len(gt_boxes)
                precision = cum_tp / (cum_tp + cum_fp)

                # Calculate AP using 11-point interpolation
                ap = 0.0
                for t in np.arange(0, 1.1, 0.1):
                    if np.sum(recall >= t) == 0:
                        p = 0.0
                    else:
                        p = np.max(precision[recall >= t])
                    ap += p / 11.0

                # Calculate final metrics
                final_precision = float(precision[-1]) if len(precision) > 0 else 0.0
                final_recall = float(recall[-1]) if len(recall) > 0 else 0.0
                f1 = (
                    2 * (final_precision * final_recall) / (final_precision + final_recall)
                    if (final_precision + final_recall) > 0
                    else 0.0
                )

                metrics = {
                    "precision": final_precision,
                    "recall": final_recall,
                    "f1": f1,
                    "ap": ap,
                    "map": ap,  # For single class, mAP equals AP
                }
                batch_metrics.append(metrics)

            return self._aggregate_metrics(batch_metrics)

        # Calculate number of batches
        num_batches = (len(input_data_batch) + batch_size - 1) // batch_size
        batch_indices = range(num_batches)

        # Process batches in parallel
        all_metrics = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_batch, idx) for idx in batch_indices]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating detection batches"):
                try:
                    batch_result = future.result()
                    all_metrics.append(batch_result)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

        # Aggregate metrics from all batches
        return self._aggregate_metrics(all_metrics)

    def _aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """
        Aggregate metrics from multiple batches.

        Args:
            metrics_list: List of dictionaries containing metrics

        Returns:
            Dictionary containing aggregated metrics
        """
        aggregated_metrics = {}
        for metric_name in metrics_list[0].keys():
            aggregated_metrics[metric_name] = sum(
                [metrics[metric_name] for metrics in metrics_list],
            ) / len(metrics_list)
        return aggregated_metrics

    def _calculate_iou(self, box1: list[float], box2: list[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: First box coordinates [x, y, w, h]
            box2: Second box coordinates [x, y, w, h]

        Returns:
            IoU score
        """
        # Convert to x1, y1, x2, y2 format
        box1_x1 = float(box1[0])
        box1_y1 = float(box1[1])
        box1_x2 = float(box1[0] + box1[2])
        box1_y2 = float(box1[1] + box1[3])

        box2_x1 = float(box2[0])
        box2_y1 = float(box2[1])
        box2_x2 = float(box2[0] + box2[2])
        box2_y2 = float(box2[1] + box2[3])

        # Calculate intersection coordinates
        x1 = max(box1_x1, box2_x1)
        y1 = max(box1_y1, box2_y1)
        x2 = min(box1_x2, box2_x2)
        y2 = min(box1_y2, box2_y2)

        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union = box1_area + box2_area - intersection

        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0
        return float(iou)
