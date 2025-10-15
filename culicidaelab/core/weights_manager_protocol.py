"""Protocol for managing machine learning model weights.

This module defines the `WeightsManagerProtocol`, which establishes an interface
for any class that manages access to model weight files. This protocol ensures loose
coupling between system components while providing a standardized way to handle model
weights across different predictors and backend types.

Example:
    Here's how to implement and use a concrete weights manager:

    ```python
    from pathlib import Path

    class LocalWeightsManager:
        def __init__(self, weights_dir: str = "model_weights"):
            self.weights_dir = Path(weights_dir)
            self.weights_dir.mkdir(exist_ok=True)

        def ensure_weights(self, predictor_type: str, backend_type: str) -> Path:
            weight_path = self.weights_dir / f"{predictor_type}_{backend_type}.pth"
            if not weight_path.exists():
                # Download or generate weights here
                pass
            return weight_path.absolute()

    # Usage
    weights_manager = LocalWeightsManager()
    model_weights_path = weights_manager.ensure_weights("classifier", "fastai")
    ```
"""

from pathlib import Path
from typing import Protocol


class WeightsManagerProtocol(Protocol):
    def ensure_weights(self, predictor_type: str, backend_type: str) -> Path:
        """Ensures model weights are available locally and returns their path.

        This method is responsible for managing model weight files, including checking
        their existence, downloading if necessary, and providing the absolute path to
        the weights file. It abstracts away the details of weight file management from
        the rest of the system.

        Args:
            predictor_type (str): The type of predictor requiring the weights.
                Common values include 'classifier', 'detector', or 'segmenter'.
            backend_type (str): The backend framework for which the weights are needed.
                Examples include 'fastai', 'onnx', 'yolo', or 'sam'.

        Returns:
            Path: Absolute path to the model weights file. The returned path is
                guaranteed to exist and be accessible.

        Example:
            ```python
            from your_module import WeightsManager

            weights_manager = WeightsManager()

            # Get weights for a FastAI classifier
            classifier_weights = weights_manager.ensure_weights(
                predictor_type="classifier",
                backend_type="fastai"
            )

            # Use the weights in a model
            model.load_state_dict(torch.load(classifier_weights))
            ```

        Note:
            Implementations should handle various scenarios such as:
            - Checking if weights exist locally
            - Downloading weights from remote sources if needed
            - Validating weight file integrity
            - Managing weight file versions
            - Handling download failures and retry logic
        """
        ...
