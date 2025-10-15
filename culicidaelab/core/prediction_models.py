"""Pydantic models for prediction outputs.

This module defines the data structures for the outputs of the different
inference predictors (detection, classification, segmentation). These models
ensure that the prediction results are well-defined and validated.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# --- Detection Models ---
class BoundingBox(BaseModel):
    """Represents a single bounding box with coordinates.

    Attributes:
        x1 (float): The top-left x-coordinate of the bounding box.
        y1 (float): The top-left y-coordinate of the bounding box.
        x2 (float): The bottom-right x-coordinate of the bounding box.
        y2 (float): The bottom-right y-coordinate of the bounding box.
    """

    x1: float = Field(..., description="Top-left x-coordinate")
    y1: float = Field(..., description="Top-left y-coordinate")
    x2: float = Field(..., description="Bottom-right x-coordinate")
    y2: float = Field(..., description="Bottom-right y-coordinate")

    def to_numpy(self) -> np.ndarray:
        """Converts the bounding box to a NumPy array.

        Returns:
            np.ndarray: A NumPy array of shape (4,) in the format [x1, y1, x2, y2].
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])


class Detection(BaseModel):
    """Represents a single detected object, including its bounding box and confidence.

    Attributes:
        box (BoundingBox): The bounding box of the detected object.
        confidence (float): The confidence score of the prediction, between 0.0 and 1.0.
    """

    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")


class DetectionPrediction(BaseModel):
    """Represents the output of a detection model for a single image.

    Attributes:
        detections (list[Detection]): A list of all objects detected in the image.
    """

    detections: list[Detection]


# --- Classification Models ---
class Classification(BaseModel):
    """Represents a single classification result with species name and confidence.

    Attributes:
        species_name (str): The predicted species name.
        confidence (float): The confidence score of the prediction, between 0.0 and 1.0.
    """

    species_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")


class ClassificationPrediction(BaseModel):
    """Represents the full output of a classification model for a single image.

    The predictions are typically sorted by confidence in descending order.

    Attributes:
        predictions (list[Classification]): A list of classification results.
    """

    predictions: list[Classification]

    def top_prediction(self) -> Classification | None:
        """Returns the top prediction (the one with the highest confidence).

        Returns:
            Classification | None: The top classification result, or None if there
            are no predictions.
        """
        return self.predictions[0] if self.predictions else None


# --- Segmentation Models ---
class SegmentationPrediction(BaseModel):
    """Represents the output of a segmentation model for a single image.

    Attributes:
        mask (np.ndarray): A 2D NumPy array (H, W) representing the binary
            segmentation mask, where non-zero values indicate the segmented object.
        pixel_count (int): The total number of positive (masked) pixels in the mask.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: np.ndarray = Field(..., description="Binary segmentation mask as a NumPy array (H, W)")
    pixel_count: int = Field(..., description="Number of positive (masked) pixels")
