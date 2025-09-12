from typing import Any
import onnxruntime
import numpy as np
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend


class ONNXBackend(BaseInferenceBackend[np.ndarray, np.ndarray]):
    def __init__(self, weights_manager: WeightsManagerProtocol):
        self.weights_manager = weights_manager
        self.session = None

    def load_model(self, predictor_type: str, **kwargs: Any):
        # 1. Backend resolves its OWN path via the manager
        model_path = self.weights_manager.resolve_weights_path(
            predictor_type=predictor_type,
            backend_type="onnx",  # The backend knows its own type
        )
        # 2. Backend loads the model from the resolved path
        self.session = onnxruntime.InferenceSession(str(model_path))

    def predict(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self.session:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        return self.session.run([output_name], {input_name: input_data})[0]  # type: ignore

    def unload_model(self):
        self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None
