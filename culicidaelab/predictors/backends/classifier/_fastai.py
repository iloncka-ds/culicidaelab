# _fastai.py
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
    """A specialized FastAI backend for classification.

    This class implements the inference backend for the classifier using the
    FastAI library. It handles model loading and prediction.

    Attributes:
        predictor_type (str): The type of predictor, which is 'classifier'.
        weights_manager (WeightsManagerProtocol): An object to manage model weights.
        model: The loaded FastAI learner model.
    """

    def __init__(self, weights_manager: WeightsManagerProtocol):
        """Initializes the ClassifierFastAIBackend.

        Args:
            weights_manager: An object that conforms to the
                WeightsManagerProtocol, used to get the model weights.
        """
        self.predictor_type = "classifier"
        super().__init__(predictor_type=self.predictor_type)
        self.weights_manager = weights_manager
        self.model = None

    def load_model(self, **kwargs: Any):
        """Loads the FastAI classifier model.

        This method retrieves the model weights path using the weights manager
        and loads the FastAI learner. It uses a workaround for loading models
        trained on POSIX systems on Windows.

        Args:
            **kwargs: Additional keyword arguments (not used).
        """
        model_path = self.weights_manager.ensure_weights(
            predictor_type=self.predictor_type,
            backend_type="torch",
        )
        with set_posix_windows():
            self.model = load_learner(model_path)

    def predict(self, input_data: Image.Image, **kwargs: Any) -> np.ndarray:
        """Performs inference on the input image.

        If the model is not already loaded, this method will load it first.
        It then uses the FastAI learner to predict the class probabilities
        for the input image.

        Args:
            input_data: The input image for classification.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            A numpy array of class probabilities.

        Raises:
            RuntimeError: If the model cannot be loaded.
        """
        if not self.model:
            try:
                self.load_model()
            except Exception as e:
                raise RuntimeError("Model is not loaded. Call load_model() first.", e)

        with set_posix_windows():
            _, _, probs = self.model.predict(input_data)  # type: ignore

        return probs.numpy()
