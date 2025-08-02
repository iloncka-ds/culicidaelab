"""Module for mosquito segmentation using the Segment Anything Model (SAM).

This module provides the MosquitoSegmenter class, which uses a pre-trained
SAM model (specifically, SAM2) to generate precise segmentation masks for
mosquitos in an image. It can be prompted with detection bounding boxes
for targeted segmentation and supports efficient batch processing.

Example:
    from culicidaelab.core.settings import Settings
    from culicidaelab.predictors import MosquitoSegmenter
    import numpy as np

    # Initialize settings and segmenter
    settings = Settings()
    segmenter = MosquitoSegmenter(settings, load_model=True)

    # --- Batch Prediction Example ---
    # Create a dummy batch of images and detection boxes
    images = [np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8) for _ in range(4)]
    # box = (center_x, center_y, width, height, confidence)
    detections_batch = [
        [(512, 512, 100, 100, 0.9)],  # Boxes for image 1
        [(256, 256, 80, 80, 0.95), (700, 700, 120, 120, 0.88)],  # Boxes for image 2
        [],  # No boxes for image 3
        [(100, 800, 50, 50, 0.92)],  # Boxes for image 4
    ]

    # Get segmentation masks for the whole batch
    masks = segmenter.predict_batch(images, detection_boxes_batch=detections_batch)

    print(f"Generated {len(masks)} masks for the batch.")
    print(f"Shape of first mask: {masks[0].shape}")

    # Clean up
    segmenter.unload_model()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeAlias, cast
from collections.abc import Sequence

import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from fastprogress.fastprogress import progress_bar


from culicidaelab.core.base_predictor import BasePredictor

from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

SegmentationPredictionType: TypeAlias = np.ndarray
SegmentationGroundTruthType: TypeAlias = np.ndarray


class MosquitoSegmenter(
    BasePredictor[np.ndarray, SegmentationPredictionType, SegmentationGroundTruthType],
):
    """Segments mosquitos in images using the SAM2 model.

    This class provides methods to load a SAM2 model, generate segmentation
    masks for entire images or specific regions defined by bounding boxes,
    and visualize the resulting masks.

    Args:
        settings (Settings): The main settings object for the library.
        load_model (bool, optional): If True, the model is loaded upon
            initialization. Defaults to False.
    """

    def __init__(self, settings: Settings, load_model: bool = False) -> None:
        """Initializes the MosquitoSegmenter."""

        weights_manager = ModelWeightsManager(
            settings=settings,
        )
        super().__init__(
            settings=settings,
            predictor_type="segmenter",
            weights_manager=weights_manager,
            load_model=load_model,
        )

    def predict(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Generates a segmentation mask for mosquitos in an image.

        If `detection_boxes` are provided in kwargs, the model will generate
        masks only for those specific regions. Otherwise, it will attempt to
        segment the most prominent object in the image.

        Args:
            input_data (np.ndarray): The input image as a NumPy array (H, W, 3).
            **kwargs (Any): Optional keyword arguments, including:
                detection_boxes (list, optional): A list of bounding boxes
                in (cx, cy, w, h, conf) format to guide segmentation.

        Returns:
            np.ndarray: A binary segmentation mask of shape (H, W), where pixels
            belonging to a mosquito are marked as 1 (or True) and 0 otherwise.

        Raises:
            RuntimeError: If the model is not loaded or fails to load.
        """
        if not self.model_loaded or self._model is None:
            self.load_model()
            if self._model is None:
                raise RuntimeError("Failed to load model")

        detection_boxes = kwargs.get("detection_boxes")

        if len(input_data.shape) == 2:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2RGB)
        elif input_data.shape[2] == 4:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_RGBA2RGB)

        model = cast(SAM2ImagePredictor, self._model)
        model.set_image(input_data)

        if detection_boxes and len(detection_boxes) > 0:
            masks = []
            for box in detection_boxes:
                x, y, w, h, _ = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                input_box = np.array([x1, y1, x2, y2])

                mask, _, _ = model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                masks.append(mask[0])
            # Combine all masks for the image with a logical OR
            combined_mask = np.logical_or.reduce([m.cpu().numpy() for m in masks])
            return combined_mask.astype(np.uint8)

        # Fallback to default prediction if no boxes are provided
        masks, _, _ = model.predict(
            point_coords=None,
            point_labels=None,
            box=None,
            multimask_output=False,
        )
        return masks[0].cpu().numpy().astype(np.uint8)

    def predict_batch(
        self,
        input_data_batch: Sequence[np.ndarray],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[SegmentationPredictionType]:
        """Generates segmentation masks for a batch of images using native batch processing.

        This method requires corresponding bounding box prompts for each image.

        Args:
            input_data_batch (Sequence[np.ndarray]): A sequence of input images.
            show_progress (bool): If True, a progress bar is displayed.
            **kwargs (Any): Optional keyword arguments, must include:
                detection_boxes_batch (Sequence[list]): A sequence of detection box lists.
                Each inner list corresponds to an image in `input_data_batch`.

        Returns:
            A list of binary segmentation masks (np.ndarray).

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If `detection_boxes_batch` length does not match image batch length.
        """
        if not self.model_loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        detection_boxes_batch = kwargs.get("detection_boxes_batch")
        if detection_boxes_batch is None:
            self._logger.warning(
                "Native batch prediction for segmenter requires 'detection_boxes_batch'. "
                "Falling back to serial prediction.",
            )
            return super().predict_batch(
                input_data_batch,
                show_progress=show_progress,
                **kwargs,
            )

        if len(input_data_batch) != len(detection_boxes_batch):
            raise ValueError(
                f"Mismatch between number of images ({len(input_data_batch)}) and "
                f"number of detection box lists ({len(detection_boxes_batch)}).",
            )

        # Prepare images and boxes
        processed_images = [
            cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if len(img.shape) == 2
            else cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            if img.shape[2] == 4
            else img
            for img in input_data_batch
        ]

        boxes_batch_np = []
        for box_list in detection_boxes_batch:
            np_boxes = [[int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)] for x, y, w, h, _ in box_list]
            boxes_batch_np.append(
                np.array(np_boxes) if np_boxes else np.empty((0, 4)),
            )

        model = cast(SAM2ImagePredictor, self._model)
        model.set_image_batch(processed_images)

        masks_batch, _, _ = model.predict_batch(
            point_coords_batch=None,
            point_labels_batch=None,
            box_batch=boxes_batch_np,
            multimask_output=False,
        )

        final_masks = []
        iterator = masks_batch
        if show_progress:
            iterator = progress_bar(
                masks_batch,
                total=len(processed_images),
                comment="Processing batch masks",
            )

        for i, masks_for_image in enumerate(iterator):
            if masks_for_image.shape[0] > 0:
                combined_mask = np.logical_or.reduce(masks_for_image.cpu().numpy())
                final_masks.append(combined_mask.astype(np.uint8))
            else:
                h, w, _ = processed_images[i].shape
                final_masks.append(np.zeros((h, w), dtype=np.uint8))

        return final_masks

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: SegmentationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Overlays a segmentation mask on the original image.

        Args:
            input_data (np.ndarray): The original image.
            predictions (SegmentationPredictionType): The binary segmentation mask
                from the `predict` method.
            save_path (str | Path | None, optional): If provided, the visualized
                image is saved to this path. Defaults to None.

        Returns:
            np.ndarray: A new image array with the segmentation mask visualized
            as a colored overlay.
        """
        if len(input_data.shape) == 2:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_GRAY2BGR)

        colored_mask = np.zeros_like(input_data)
        overlay_color_bgr = str_to_bgr(self.config.visualization.overlay_color)
        colored_mask[predictions > 0] = np.array(overlay_color_bgr)

        overlay = cv2.addWeighted(
            input_data,
            1,
            colored_mask,
            self.config.visualization.alpha,
            0,
        )

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return overlay

    def _evaluate_from_prediction(
        self,
        prediction: SegmentationPredictionType,
        ground_truth: SegmentationGroundTruthType,
    ) -> dict[str, float]:
        """Calculates segmentation metrics for a single predicted mask.

        Computes Intersection over Union (IoU), precision, recall, and F1-score.

        Args:
            prediction (SegmentationPredictionType): The predicted binary mask.
            ground_truth (SegmentationGroundTruthType): The ground truth binary mask.

        Returns:
            dict[str, float]: A dictionary containing the segmentation metrics.
        """
        prediction = prediction.astype(bool)
        ground_truth = ground_truth.astype(bool)

        intersection = np.logical_and(prediction, ground_truth)
        union = np.logical_or(prediction, ground_truth)

        intersection_sum = np.sum(intersection)
        prediction_sum = np.sum(prediction)
        ground_truth_sum = np.sum(ground_truth)
        union_sum = np.sum(union)

        iou = intersection_sum / union_sum if union_sum > 0 else 0.0
        precision = intersection_sum / prediction_sum if prediction_sum > 0 else 0.0
        recall = intersection_sum / ground_truth_sum if ground_truth_sum > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _load_model(self) -> None:
        """Loads the SAM2 model, selecting the best device and optimizing for performance.

        Raises:
            ValueError: If the model configuration path is missing.
            RuntimeError: If the model fails to load.
        """
        # 1. Select best available device
        device_str = self.config.device or "auto"
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(device_str)

        self._logger.info(f"Using device: {device}")

        # 2. Apply performance enhancements for the selected device
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # Enable TF32 for Ampere and newer GPUs
            if torch.cuda.get_device_properties(0).major >= 8:
                self._logger.info("Enabling TF32 for Ampere+ GPU for better performance.")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            self._logger.warning(
                "Support for MPS devices is preliminary. SAM2 may give numerically "
                "different outputs or have degraded performance on MPS.",
            )

        # 3. Build and load the model
        model_config_path = getattr(self.config, "model_config_path", None)
        if not model_config_path:
            raise ValueError(
                "Missing required configuration: 'model_config_path' must be set for the segmenter.",
            )

        try:
            self._logger.info(f"Building SAM2 model from config: {model_config_path}")
            sam2_model = build_sam2(model_config_path, str(self.model_path), device=device)
            self._model = SAM2ImagePredictor(sam2_model)
            self._model_loaded = True
            self._logger.info("SAM2 model and predictor loaded successfully.")
        except Exception as e:
            self._model = None
            raise RuntimeError(
                f"Failed to load SAM model from {self.model_path}. "
                f"Please check model path and configuration. Error: {e}",
            ) from e
