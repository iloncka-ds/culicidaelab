from pathlib import Path
from typing import Protocol


class WeightsManagerProtocol(Protocol):
    """
    Defines the interface for any class that manages model weights.

    This protocol ensures that the core components can work with any
    weights manager without depending on its concrete implementation.
    """

    def ensure_weights(self, predictor_type: str) -> Path:
        """
        Ensures weights for a given predictor type are available locally.

        Args:
            predictor_type (str): The key for the predictor (e.g., 'classifier').

        Returns:
            Path: The local path to the model weights file.
        """
        ...
