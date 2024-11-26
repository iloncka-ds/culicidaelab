"""
Module for mosquito detection in images using YOLOv8.
"""

from __future__ import annotations

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Any
import torch


class MosquitoDetector:
    """Class for detecting mosquitos in images using YOLOv8."""

    def __init__(self, model_path: str = "", confidence_threshold: float = 0.5):
        """
        Initialize the mosquito detector.

        Args:
            model_path (str): Path to pre-trained YOLOv8 model weights
            confidence_threshold (float): Minimum confidence score for detection
        """
        if not model_path:
            raise ValueError("model_path must be provided for YOLOv8 model")

        self.confidence_threshold = confidence_threshold
        self.model = YOLO(model_path)

    def detect(self, image: str | np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """
        Detect mosquitos in an image.

        Args:
            image: Input image (file path or numpy array)

        Returns:
            List of tuples (x, y, width, height, confidence)
        """
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)

        # Process predictions
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates and confidence
                xyxy = box.xyxy  # Shape: (1, 4)
                if isinstance(xyxy, torch.Tensor):
                    x1, y1, x2, y2 = xyxy[0].cpu().numpy()
                else:
                    x1, y1, x2, y2 = xyxy[0]  # Handle case where it's already numpy
                conf = float(box.conf[0])

                # Convert to center format
                w = x2 - x1
                h = y2 - y1
                x = x1 + w / 2
                y = y1 + h / 2

                detections.append((x, y, w, h, conf))

        return detections

    def visualize_detections(self, image: str | np.ndarray, detections: list[tuple]) -> np.ndarray:
        """
        Draw bounding boxes around detected mosquitos.

        Args:
            image: Input image
            detections: List of detection tuples (x, y, width, height, confidence)

        Returns:
            np.ndarray: Image with drawn detections
        """
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            image = image.copy()

        for x, y, w, h, conf in detections:
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Mosquito: {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        return image

    def train(self, data_yaml: str, epochs: int = 100, batch_size: int = 16):
        """
        Train the YOLOv8 model on custom dataset.

        Args:
            data_yaml: Path to data configuration file
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device="cuda:0",  # Use GPU if available
        )

    def evaluate(
        self,
        true_boxes: list[list[float | int]],
        pred_boxes: list[list[float | int]],
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Evaluate object detection model performance.

        Args:
            true_boxes: List of ground truth bounding boxes [x, y, w, h]
            pred_boxes: List of predicted bounding boxes [x, y, w, h, conf]
            iou_threshold: IoU threshold for considering a detection correct

        Returns:
            Dictionary containing metrics (precision, recall, mAP, etc.)
        """
        # Handle empty cases
        if not true_boxes and not pred_boxes:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "map": 0.0,
            }

        if not true_boxes:  # No ground truth boxes
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "map": 0.0,
            }

        if not pred_boxes:  # No predicted boxes
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap": 0.0,
                "map": 0.0,
            }

        # Sort predictions by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        # Initialize metrics
        tp = np.zeros(len(pred_boxes), dtype=np.float32)
        fp = np.zeros(len(pred_boxes), dtype=np.float32)
        used_gt = set()

        # Calculate IoU for each prediction
        for i, pred in enumerate(pred_boxes):
            max_iou = 0.0
            max_idx = -1

            for j, gt in enumerate(true_boxes):
                if j in used_gt:
                    continue

                # Calculate IoU
                iou = self.calculate_iou(pred[:4], gt)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j

            # Check if detection is correct
            if max_iou >= iou_threshold:
                tp[i] = 1.0
                used_gt.add(max_idx)
            else:
                fp[i] = 1.0

        # Calculate cumulative metrics
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / len(true_boxes)
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
        final_precision = float(precision[-1])
        final_recall = float(recall[-1])
        f1 = (
            2 * (final_precision * final_recall) / (final_precision + final_recall)
            if (final_precision + final_recall) > 0
            else 0.0
        )

        return {
            "precision": final_precision,
            "recall": final_recall,
            "f1": float(f1),
            "ap": float(ap),
            "map": float(ap),  # mAP is same as AP for single class
        }

    def calculate_iou(self, box1: list[float | int], box2: list[float | int]) -> float:
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
