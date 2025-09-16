from typing import Any, cast
import onnxruntime
from PIL import Image
import numpy as np
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.config_models import PredictorConfig
from culicidaelab.core.base_inference_backend import BaseInferenceBackend


class ClassifierONNXBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    def __init__(
        self,
        weights_manager: WeightsManagerProtocol,
        config: PredictorConfig,
    ):
        super().__init__(predictor_type="classifier")
        self.weights_manager = weights_manager
        self.config = config
        self.session = None

    def load_model(self, **kwargs: Any):
        # 1. Backend resolves its OWN path via the manager
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="onnx",  # The backend knows its own type
        )
        # 2. Backend loads the model from the resolved path
        self.session = onnxruntime.InferenceSession(str(model_path))

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
        if not self.session:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        preprocessed_data = self._preprocess(input_data)
        model_outputs = self.session.run([output_name], {input_name: preprocessed_data})[0]  # type: ignore
        logits_array = cast(list[Any], model_outputs)[0]
        final_result = self._postprocess(logits_array)
        return final_result

    def unload_model(self):
        self.session = None

    @property
    def is_loaded(self) -> bool:
        return self.session is not None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesses the input PIL Image to the format expected by the ONNX model.

        This involves resizing, converting to a float tensor, normalizing,
        and adding a batch dimension.
        """
        # 1. Get preprocessing parameters from the configuration
        params = self.config.params
        input_size = params["input_size"]
        mean = np.array(params["mean"], dtype=np.float32)
        std = np.array(params["std"], dtype=np.float32)

        # 2. Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 3. Resize the image
        image_resized = image.resize((input_size, input_size))

        # 4. Convert to NumPy array, change data type to float32, and scale pixels to [0, 1]
        # THIS IS THE KEY FIX FOR THE DATA TYPE ERROR
        input_array = np.array(image_resized, dtype=np.float32) / 255.0

        # 5. Normalize the tensor using the specified mean and std
        normalized_array = (input_array - mean) / std

        # 6. Transpose dimensions from (Height, Width, Channels) to (Channels, Height, Width)
        transposed_array = normalized_array.transpose((2, 0, 1))

        # 7. Add a batch dimension to create a final shape of (1, C, H, W)
        batch_array = np.expand_dims(transposed_array, axis=0)

        return batch_array

    def _postprocess(self, logits: np.ndarray) -> np.ndarray:
        """
        Postprocesses the output of the ONNX model to get probabilities.
        """
        return self._softmax(logits)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """
        Computes softmax probabilities from a 1D array of logits.
        This implementation is numerically stable to prevent overflow.
        """
        # Subtract the maximum logit for numerical stability
        exp_logits = np.exp(logits - np.max(logits))
        # Normalize to get probabilities
        return exp_logits / np.sum(exp_logits)
