"""Protocol for managing model weights.

This module defines the `WeightsManagerProtocol`, which establishes an interface
for any class that manages access to model weight files. This ensures loose
coupling between system components.
"""

from pathlib import Path
from typing import Protocol


class WeightsManagerProtocol(Protocol):
    def ensure_weights(self, predictor_type: str, backend_type: str) -> Path:
        """
        Ensures weights for a given predictor and backend type are available locally,
        downloading them if necessary, and returns the absolute path.
        """
        ...
