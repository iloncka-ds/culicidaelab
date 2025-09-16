from typing import Any
from ultralytics import YOLO
from PIL import Image
import numpy as np
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend


class DetectorYOLOBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    def __init__(self, weights_manager: WeightsManagerProtocol):
        super().__init__(predictor_type="detector")
        self.weights_manager = weights_manager
        self.model = None

    def load_model(self, **kwargs: Any):
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )
        self.model = YOLO(str(model_path))

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
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
