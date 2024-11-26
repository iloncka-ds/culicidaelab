"""
Module for mosquito segmentation using SAM (Segment Anything Model).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from typing import Any

# Third Party


class MosquitoSegmenter:
    """Class for segmenting mosquitos in images using SAM."""

    def __init__(self, model_type: str = "vit_h", checkpoint_path: str = "") -> None:
        """
        Initialize the mosquito segmenter.

        Args:
            model_type (str): SAM model type ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path (str): Path to SAM model checkpoint
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize SAM
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be provided for SAM model")

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def segment(
        self,
        image: str | np.ndarray,
        detection_boxes: list[tuple[float, float, float, float, float]] | None = None,
    ) -> np.ndarray:
        """
        Segment mosquitos in an image.

        Args:
            image: Input image (file path or numpy array)
            detection_boxes: Optional list of detection boxes (x, y, w, h, conf)

        Returns:
            np.ndarray: Binary mask of segmented mosquitos
        """
        # Load and prepare image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image in predictor
        self.predictor.set_image(image)

        # If detection boxes are provided, use them as prompts
        if detection_boxes:
            masks = []
            for x, y, w, h, _ in detection_boxes:
                # Convert center format to box format
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Get input box
                input_box = np.array([x1, y1, x2, y2])

                # Generate mask
                masks_chunk, _, _ = self.predictor.predict(
                    box=input_box,
                    multimask_output=False,
                )
                masks.append(masks_chunk[0])

            # Combine all masks
            if masks:
                final_mask = np.logical_or.reduce(masks)
            else:
                final_mask = np.zeros(image.shape[:2], dtype=bool)
        else:
            # If no boxes provided, use automatic mask generation
            masks = self.predictor.generate()
            final_mask = np.logical_or.reduce([mask for mask in masks])

        return final_mask.astype(np.uint8)

    def apply_mask(
        self,
        image: str | np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Apply segmentation mask to original image.

        Args:
            image: Original image
            mask: Binary segmentation mask

        Returns:
            np.ndarray: Image with highlighted segmentation
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create colored overlay
        overlay = np.zeros_like(image)
        overlay[mask == 1] = [0, 255, 0]  # Green overlay for segmented regions

        # Blend original image with overlay
        alpha = 0.3
        output = cv2.addWeighted(image, 1, overlay, alpha, 0)

        return output

    def evaluate(
        self,
        true_masks: list[np.ndarray],
        pred_masks: list[np.ndarray],
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Evaluate segmentation model performance.

        Args:
            true_masks: List of ground truth binary masks
            pred_masks: List of predicted binary masks
            iou_threshold: IoU threshold for considering a segmentation correct

        Returns:
            Dictionary containing metrics (IoU, Dice coefficient, etc.)
        """
        if not true_masks or not pred_masks:
            return {
                "mean_iou": 0.0,
                "mean_dice": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        if len(true_masks) != len(pred_masks):
            raise ValueError("Number of true and predicted masks must match")

        # Initialize metrics
        ious = []
        dices = []
        precisions = []
        recalls = []

        for true_mask, pred_mask in zip(true_masks, pred_masks):
            if true_mask.shape != pred_mask.shape:
                raise ValueError("Mask shapes must match")

            # Calculate intersection and union
            intersection = np.logical_and(true_mask, pred_mask).sum()
            union = np.logical_or(true_mask, pred_mask).sum()
            true_positives = intersection
            false_positives = pred_mask.sum() - intersection
            false_negatives = true_mask.sum() - intersection

            # Calculate IoU
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

            # Calculate Dice coefficient
            dice = (
                2 * intersection / (true_mask.sum() + pred_mask.sum())
                if (true_mask.sum() + pred_mask.sum()) > 0
                else 0.0
            )
            dices.append(dice)

            # Calculate precision and recall
            precision = (
                true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            )
            recall = (
                true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            )
            precisions.append(precision)
            recalls.append(recall)

        # Calculate mean metrics
        mean_iou = float(np.mean(ious))
        mean_dice = float(np.mean(dices))
        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))

        # Calculate F1 score
        f1 = (
            2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
            if (mean_precision + mean_recall) > 0
            else 0.0
        )

        return {
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": float(f1),
            "per_mask_iou": [float(iou) for iou in ious],
            "per_mask_dice": [float(dice) for dice in dices],
        }
