from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# --- Detection Models ---
class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left x-coordinate")
    y1: float = Field(..., description="Top-left y-coordinate")
    x2: float = Field(..., description="Bottom-right x-coordinate")
    y2: float = Field(..., description="Bottom-right y-coordinate")

    def to_numpy(self) -> np.ndarray:
        """Converts the bounding box to a NumPy array of shape (4,).

        Returns:
            np.ndarray: A NumPy array in the format [x1, y1, x2, y2].
        """
        return np.array([self.x1, self.y1, self.x2, self.y2])


class Detection(BaseModel):
    box: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")


class DetectionPrediction(BaseModel):
    detections: list[Detection]


# --- Classification Models ---
class Classification(BaseModel):
    species_name: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")


class ClassificationPrediction(BaseModel):
    predictions: list[Classification]

    def top_prediction(self) -> Classification | None:
        return self.predictions[0] if self.predictions else None


# --- Segmentation Models ---
class SegmentationPrediction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    mask: np.ndarray = Field(..., description="Binary segmentation mask as a NumPy array (H, W)")
    pixel_count: int = Field(..., description="Number of positive (masked) pixels")
