"""YOLO backend for the detector."""

from typing import Any
from ultralytics import YOLO
from PIL import Image
import numpy as np
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend
from culicidaelab.core.config_models import PredictorConfig


class DetectorYOLOBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    """YOLO backend for mosquito object detection.

    This class implements the inference backend using `ultralytics.YOLO`.
    It handles loading the model from a `.pt` file, moving it to the
    configured device (`cpu` or `cuda`), and running predictions. The output
    is standardized to a NumPy array where each row is `[x1, y1, x2, y2, confidence]`.

    Attributes:
        weights_manager (WeightsManagerProtocol): An object for managing model weights.
        config (PredictorConfig): The configuration for the predictor.
        model (ultralytics.YOLO | None): The loaded YOLO model object.

    """

    def __init__(self, weights_manager: WeightsManagerProtocol, config: PredictorConfig):
        """Initializes the DetectorYOLOBackend.

        Args:
            weights_manager (WeightsManagerProtocol): An object for managing model weights.
            config (PredictorConfig): The configuration for the predictor, which
                includes the target `device`.
        """
        super().__init__(predictor_type="detector")
        self.weights_manager = weights_manager
        self.config = config
        self.model = None

    def load_model(self, **kwargs: Any):
        """Loads the YOLO detector model and moves it to the configured device."""
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )

        self.model = YOLO(str(model_path))
        self.model.to(self.config.device)

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
        """Performs inference on a single input image.

        Args:
            input_data (Image.Image): The input image for detection.
            **kwargs: Additional keyword arguments passed to the YOLO model's
                prediction call (e.g., `conf`, `iou`).

        Returns:
            np.ndarray: A numpy array of detections with shape (N, 5), where
            each row is `[x1, y1, x2, y2, confidence]`. Returns an empty
            array if no objects are detected.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        results = self.model(source=input_data, **kwargs)

        if not results:
            return np.array([])
        result = results[0]
        boxes = result.boxes.cpu().numpy()
        if len(boxes) == 0:
            return np.array([])
        return np.hstack((boxes.xyxy, boxes.conf.reshape(-1, 1)))

    def predict_batch(
        self,
        input_data_batch: list[Image.Image],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Performs inference on a batch of input images.

        Args:
            input_data_batch (list[np.ndarray]): A list of input images.
            show_progress (bool): If True, a progress bar is shown (not used by YOLO).
            **kwargs: Additional keyword arguments passed to the YOLO model.

        Returns:
            list[np.ndarray]: A list of numpy arrays, where each array contains
            the detections for the corresponding input image.

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
