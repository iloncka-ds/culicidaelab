import logging
from ultralytics import SAM
import numpy as np
import torch
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend

logger = logging.getLogger(__name__)


class SegmenterSAMBackend(BaseInferenceBackend[np.ndarray, np.ndarray]):
    def __init__(self, weights_manager: WeightsManagerProtocol):
        super().__init__(predictor_type="segmenter")
        self.weights_manager = weights_manager
        self.model = None

    def load_model(self, **kwargs):
        model_path = self.weights_manager.resolve_weights_path(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )
        device = kwargs.get("device", "")
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAM(str(model_path))
        if self.model:
            self.model.to(device)

    def predict(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        h, w = input_data.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        detection_boxes = kwargs.get("detection_boxes", [])
        points = kwargs.get("points", [])
        labels = kwargs.get("labels", [])
        verbose = kwargs.get("verbose", False)
        model_prompts = {}

        if detection_boxes is not None and len(detection_boxes) > 0:
            first_box = detection_boxes[0]
            if len(first_box) == 5:  # Potentially box with confidence score
                boxes_xyxy = [box[:4] for box in detection_boxes]
            elif len(first_box) == 4:
                boxes_xyxy = detection_boxes
            else:
                logger.warning(
                    "Invalid format for detection_boxes.",
                    f"Expected 4 or 5 elements, got {len(first_box)}. Ignoring boxes.",
                )
                boxes_xyxy = []

            if len(boxes_xyxy) > 0:
                logger.debug(f"Using {len(boxes_xyxy)} detection boxes for segmentation.")
                model_prompts["bboxes"] = boxes_xyxy

        if points is not None and len(points) > 0:
            if labels is None:
                raise ValueError("'labels' must be provided when 'points' are given.")

            # Normalize single point/label to a list of lists for consistent processing
            is_single_point = isinstance(points[0], (int, float))
            if is_single_point:
                points = [points]
                # Also ensure label is a list
                if not isinstance(labels, list):
                    labels = [labels]

            if len(points) != len(labels):
                raise ValueError(
                    f"Mismatch between number of points ({len(points)}) and " f"labels ({len(labels)}).",
                )
            logger.debug(f"Using {len(points)} points for segmentation.")
            model_prompts["points"] = points
            model_prompts["labels"] = labels

        if not model_prompts:
            message = "No valid prompts (boxes, points) provided; returning empty mask."
            logger.debug(message)
            return empty_mask

        results = self.model(input_data, verbose=verbose, **model_prompts)
        if not results:
            return empty_mask

        masks_np = results[0].masks.data.cpu().numpy()  # type: ignore
        if masks_np.shape[0] > 0:
            # Combine masks with a logical OR
            return np.logical_or.reduce(masks_np).astype(np.uint8)
        else:
            return empty_mask
