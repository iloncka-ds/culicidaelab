"""FastAI backend for the classifier."""

from typing import Any
from fastai.learner import load_learner
import numpy as np
from PIL import Image

import platform
import pathlib
from contextlib import contextmanager
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend
from culicidaelab.core.config_models import PredictorConfig


@contextmanager
def set_posix_windows():
    """Temporarily patch pathlib for Windows FastAI model loading.

    FastAI models saved on a POSIX system (like Linux or macOS) and loaded on
    Windows can cause `pathlib` errors. This context manager temporarily
    aliases `pathlib.PosixPath` to `pathlib.WindowsPath` to work around this
    issue, ensuring that the model loads correctly on Windows.

    Yields:
        None: This context manager does not return a value.
    """
    if platform.system() == "Windows":
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
    else:
        yield


class ClassifierFastAIBackend(BaseInferenceBackend[Image.Image, np.ndarray]):
    """FastAI backend for mosquito species classification.

    This class implements the inference backend using a `fastai.learner.Learner`.
    It handles loading the model from a pickle file and moving it to the
    device specified in the configuration (`cpu` or `cuda`). It includes a
    workaround for loading models trained on POSIX systems on Windows.

    Attributes:
        weights_manager (WeightsManagerProtocol): An object for managing model weights.
        config (PredictorConfig): The configuration for the predictor.
        model (fastai.learner.Learner | None): The loaded FastAI learner object.

    """

    def __init__(self, weights_manager: WeightsManagerProtocol, config: PredictorConfig):
        """Initializes the ClassifierFastAIBackend.

        Args:
            weights_manager (WeightsManagerProtocol): An object for managing model weights.
            config (PredictorConfig): The configuration for the predictor, which
                includes the target `device`.
        """
        super().__init__(predictor_type="classifier")
        self.weights_manager = weights_manager
        self.config = config
        self.model = None

    def load_model(self, **kwargs: Any):
        """Loads the FastAI classifier model and moves it to the configured device."""
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )
        with set_posix_windows():
            self.model = load_learner(model_path)

        self.model.to(self.config.device)

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
        """Performs inference on the input image.

        Args:
            input_data (Image.Image): The input image for classification.
            **kwargs: Additional arguments (not used).

        Returns:
            np.ndarray: A numpy array of class probabilities.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        with set_posix_windows():
            _, _, probs = self.model.predict(input_data)

        return probs.numpy()
