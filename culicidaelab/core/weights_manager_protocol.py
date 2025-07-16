"""Protocol for managing model weights.

This module defines the `WeightsManagerProtocol`, which establishes an interface
for any class that manages access to model weight files. This ensures loose
coupling between system components.
"""

from pathlib import Path
from typing import Protocol


class WeightsManagerProtocol(Protocol):
    """Defines the interface for any class that manages model weights.

    This protocol ensures that core components can work with any weights
    manager without depending on its concrete implementation.
    """

    def ensure_weights(self, predictor_type: str) -> Path:
        """Ensures weights for a given predictor type are available locally.

        This method might download the weights if they are missing or simply
        return the path if they already exist.

        Args:
            predictor_type (str): The key for the predictor (e.g., 'classifier').

        Returns:
            Path: The local path to the model weights file.
        """
        ...
