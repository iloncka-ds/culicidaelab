from typing import Any
from fastai.learner import load_learner
import numpy as np

import platform
import pathlib
from contextlib import contextmanager
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol
from culicidaelab.core.base_inference_backend import BaseInferenceBackend


@contextmanager
def set_posix_windows():
    """Temporarily patch pathlib for Windows FastAI model loading."""
    if platform.system() == "Windows":
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
    else:
        yield


class FastAIBackend(BaseInferenceBackend[np.ndarray, np.ndarray]):
    def __init__(
        self,
        weights_manager: WeightsManagerProtocol,
    ):
        self.weights_manager = weights_manager
        self.model = None

    def load_model(self, predictor_type: str = "classifier", **kwargs: Any):
        model_path = self.weights_manager.resolve_weights_path(
            predictor_type=predictor_type,
            backend_type="torch",
        )
        with set_posix_windows():
            self.model = load_learner(model_path)

    def predict(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self.model:
            try:
                self.load_model()  # as default it loads classifier
            except Exception as e:
                raise RuntimeError("Model is not loaded. Call load_model() first.", e)

        with set_posix_windows():
            _, _, probs = self.model.predict(input_data)  # type: ignore

        return np.ndarray(probs)
