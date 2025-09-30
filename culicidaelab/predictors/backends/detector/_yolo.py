"""YOLO backend for the detector."""

from typing import Any
from ultralytics import YOLO
from PIL import Image
import numpy as np
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend


class DetectorYOLOBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    """A specialized YOLO backend for object detection.

    This class implements the inference backend for the detector using the
    YOLO (You Only Look Once) model from Ultralytics. It handles model loading
    and prediction for both single images and batches of images.

    Attributes:
        predictor_type (str): The type of predictor, which is 'detector'.
        weights_manager (WeightsManagerProtocol): An object to manage model weights.
        model (YOLO | None): The loaded YOLO model.
    """

    def __init__(self, weights_manager: WeightsManagerProtocol):
        """Initializes the DetectorYOLOBackend.

        Args:
            weights_manager: An object that conforms to the
                WeightsManagerProtocol, used to get the model weights.
        """
        super().__init__(predictor_type="detector")
        self.weights_manager = weights_manager
        self.model = None

    def load_model(self, **kwargs: Any):
        """Loads the YOLO detector model.

        This method retrieves the model weights path using the weights manager
        and loads the YOLO model.

        Args:
            **kwargs: Additional keyword arguments (not used).
        """
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )
        self.model = YOLO(str(model_path))

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
        """Performs inference on a single input image.

        If the model is not already loaded, this method will raise a RuntimeError.
        It runs the YOLO model on the input image and returns the bounding boxes
        and confidence scores.

        Args:
            input_data: The input image for detection.
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            A numpy array of detected bounding boxes, where each row is
            [x1, y1, x2, y2, confidence]. Returns an empty array if no
            objects are detected.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        results = self.model(source=input_data)

        # This backend is used for both detection and segmentation.
        # We need to standardize the output based on the result type.
        if not results:
            return np.array([])

        result = results[0]

        boxes = result.boxes.cpu().numpy()
        # Combine xyxy and confidence into a single array
        if len(boxes) == 0:
            return np.array([])
        # Assumes boxes have xyxy and conf, which is standard for ultralytics
        return np.hstack((boxes.xyxy, boxes.conf.reshape(-1, 1)))

    def predict_batch(
        self,
        input_data_batch: list[np.ndarray],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Performs inference on a batch of input images.

        If the model is not already loaded, this method will raise a RuntimeError.
        It runs the YOLO model on the batch of images and returns the
        bounding boxes and confidence scores for each image.

        Args:
            input_data_batch: A list of input images for detection.
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            A list of numpy arrays, where each array contains the detected
            bounding boxes for the corresponding input image. Each row in
            the array is [x1, y1, x2, y2, confidence].

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        results_list = self.model(source=input_data_batch, **kwargs)
        standardized_outputs = []

        for result in results_list:
            boxes = result.boxes.cpu().numpy()
            if len(boxes) == 0:
                standardized_outputs.append(np.array([]))
            else:
                standardized_outputs.append(np.hstack((boxes.xyxy, boxes.conf.reshape(-1, 1))))

        return standardized_outputs
