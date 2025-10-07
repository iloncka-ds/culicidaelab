"""ONNX backend for the classifier."""

from typing import Any, cast
import onnxruntime
from PIL import Image
import numpy as np
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.config_models import PredictorConfig
from culicidaelab.core.base_inference_backend import BaseInferenceBackend


class ClassifierONNXBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    """ONNX backend for mosquito species classification.

    This class implements the inference backend for the classifier using the
    ONNX Runtime. It handles model loading, prediction, data pre/post-processing,
    and respects the device setting (`cpu` or `cuda`) from the configuration
    to select the appropriate execution provider.

    Attributes:
        weights_manager (WeightsManagerProtocol): An object to manage model weights.
        config (PredictorConfig): The configuration for the predictor.
        session (onnxruntime.InferenceSession | None): The ONNX Runtime session.
    """

    def __init__(
        self,
        weights_manager: WeightsManagerProtocol,
        config: PredictorConfig,
    ):
        """Initializes the ClassifierONNXBackend.

        Args:
            weights_manager (WeightsManagerProtocol): An object that conforms to the
                WeightsManagerProtocol, used to get the model weights.
            config (PredictorConfig): The configuration for the predictor, containing
                parameters like `device` and preprocessing values.
        """
        super().__init__(predictor_type="classifier")
        self.weights_manager = weights_manager
        self.config = config
        self.session = None

    def load_model(self, **kwargs: Any):
        """Loads the ONNX classifier model.

        This method retrieves the model weights path, checks the configured
        device, and creates an ONNX Runtime inference session with the
        appropriate execution provider (CUDA or CPU).

        Args:
            **kwargs: Additional keyword arguments may be used.
        """
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="onnx",
        )

        available_providers = onnxruntime.get_available_providers()
        providers = []

        if self.config.device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            if self.config.device == "cuda":
                print(
                    "WARNING: Device 'cuda' requested, but CUDAExecutionProvider "
                    "is not available. Falling back to CPU.",
                )

        self.session = onnxruntime.InferenceSession(str(model_path), providers=providers)

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
        """Performs inference on the input image.

        If the model is not loaded, this method will raise a RuntimeError.
        It preprocesses the image, runs inference, and postprocesses the output.

        Args:
            input_data: The input image for classification.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            np.ndarray: A numpy array of class probabilities.

        """
        if not self.session:
            self.load_model()

        input_name = self.session.get_inputs()[0].name  # type: ignore
        output_name = self.session.get_outputs()[0].name  # type: ignore
        preprocessed_data = self._preprocess(input_data)
        model_outputs = self.session.run([output_name], {input_name: preprocessed_data})[0]  # type: ignore
        logits_array = cast(list[Any], model_outputs)[0]
        final_result = self._postprocess(logits_array)
        return final_result

    def unload_model(self):
        """Unloads the model and releases resources."""
        self.session = None

    @property
    def is_loaded(self) -> bool:
        """Checks if the model is loaded."""
        return self.session is not None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Preprocesses the input PIL Image to the format expected by the ONNX model."""
        params = self.config.params
        input_size = params["input_size"]
        mean = np.array(params["mean"], dtype=np.float32)
        std = np.array(params["std"], dtype=np.float32)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_resized = image.resize((input_size, input_size))
        input_array = np.array(image_resized, dtype=np.float32) / 255.0
        normalized_array = (input_array - mean) / std
        transposed_array = normalized_array.transpose((2, 0, 1))
        batch_array = np.expand_dims(transposed_array, axis=0)
        return batch_array

    def _postprocess(self, logits: np.ndarray) -> np.ndarray:
        """Postprocesses the output of the ONNX model to get probabilities."""
        return self._softmax(logits)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Computes softmax probabilities from a 1D array of logits."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
