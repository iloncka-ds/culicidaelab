"""SAM backend for the segmenter."""

import logging
from ultralytics import SAM
from PIL import Image
import numpy as np
import torch
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend

logger = logging.getLogger(__name__)


class SegmenterSAMBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    """A specialized SAM backend for image segmentation.

    This class implements the inference backend for the segmenter using the
    Segment Anything Model (SAM) from Ultralytics. It handles model loading
    and prediction based on various prompts like bounding boxes or points.

    Attributes:
        predictor_type (str): The type of predictor, which is 'segmenter'.
        weights_manager (WeightsManagerProtocol): An object to manage model weights.
        model (SAM | None): The loaded SAM model.
    """

    def __init__(self, weights_manager: WeightsManagerProtocol):
        """Initializes the SegmenterSAMBackend.

        Args:
            weights_manager: An object that conforms to the
                WeightsManagerProtocol, used to get the model weights.
        """
        super().__init__(predictor_type="segmenter")
        self.weights_manager = weights_manager
        self.model = None

    def load_model(self, **kwargs):
        """Loads the SAM segmenter model.

        This method retrieves the model weights path using the weights manager
        and loads the SAM model. It also moves the model to the appropriate
        device (GPU if available, otherwise CPU).

        Args:
            **kwargs: Additional keyword arguments. Can include 'device' to
                specify the device to load the model on.
        """
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )
        device = kwargs.get("device", "")
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAM(str(model_path))
        if self.model:
            self.model.to(device)

    def predict(self, input_data: Image.Image, **kwargs) -> np.ndarray:
        """Performs inference on a single input image.

        If the model is not already loaded, this method will raise a RuntimeError.
        It uses prompts (bounding boxes or points) to generate segmentation masks.

        Args:
            input_data: The input image for segmentation.
            **kwargs: Additional keyword arguments.
                detection_boxes (list): A list of bounding boxes to use as prompts.
                points (list): A list of points to use as prompts.
                labels (list): A list of labels for the points (1 for foreground,
                    0 for background).
                verbose (bool): Whether to print model output.

        Returns:
            A numpy array representing the combined segmentation mask. Returns an
            empty mask if no prompts are provided or no masks are generated.

        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If points are provided without corresponding labels, or if
                the number of points and labels do not match.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        h, w = input_data.size
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
