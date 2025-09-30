"""Abstract base class for inference backends.

This module defines the interface for all inference backends. An inference backend
is responsible for loading a model and running predictions. This abstract class
provides a common structure, including a default iterative implementation for
batch prediction.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from fastprogress.fastprogress import progress_bar

logger = logging.getLogger(__name__)

InputDataType = TypeVar("InputDataType")
"""Generic type for input data to a model."""

PredictionType = TypeVar("PredictionType")
"""Generic type for the prediction output from a model."""


class BaseInferenceBackend(Generic[InputDataType, PredictionType], ABC):
    """Abstract base class for an inference backend.

    This class defines the required methods for an inference backend, which is
    responsible for loading a model and running predictions. It includes a default
    implementation for batch prediction that iterates through single predictions.

    Attributes:
        predictor_type (str): The type of predictor this backend serves (e.g., 'classifier').
        model (Any): The loaded model object. Initially None.
    """

    def __init__(
        self,
        predictor_type: str,
    ):
        """Initializes the BaseInferenceBackend.

        Args:
            predictor_type: The type of predictor (e.g., 'classifier', 'detector').
        """
        self.predictor_type = predictor_type
        self.model: Any = None

    @abstractmethod
    def load_model(self, **kwargs: Any) -> None:
        """Loads the model into memory.

        This method should handle all aspects of model loading, such as reading
        weights from a file and preparing the model for inference.

        Args:
            **kwargs: Backend-specific arguments for model loading.
        """
        ...

    @abstractmethod
    def predict(self, input_data: InputDataType, **kwargs: Any) -> PredictionType:
        """Runs a prediction on a single input.

        Args:
            input_data: The data to be processed by the model.
            **kwargs: Additional backend-specific arguments for prediction.

        Returns:
            The prediction result.
        """
        ...

    def unload_model(self) -> None:
        """Unloads the model and releases resources.

        This method is intended to free up memory (especially GPU memory) by
        deleting the model instance.
        """
        self.model = None
        logger.info(f"Model for {self.predictor_type} has been unloaded.")

    @property
    def is_loaded(self) -> bool:
        """Checks if the model is loaded into memory.

        Returns:
            True if the model is loaded, False otherwise.
        """
        return self.model is not None

    def predict_batch(
        self,
        input_data_batch: list[InputDataType],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[PredictionType]:
        """Makes predictions on a batch of inputs.

        This method provides a default implementation that iterates through the batch
        and calls `predict` for each item. Backends that support native batching
        should override this method for better performance.

        Args:
            input_data_batch: A list of inputs to process.
            show_progress: If True, displays a progress bar.
            **kwargs: Additional arguments to pass to the `predict` method.

        Returns:
            A list of prediction results.
        """
        if not input_data_batch:
            return []

        if not self.is_loaded:
            self.load_model(**kwargs)

        iterator = input_data_batch
        if show_progress:
            iterator = progress_bar(input_data_batch, total=len(input_data_batch))

        # The core logic for iterative batch prediction.
        raw_predictions = [self.predict(input_data, **kwargs) for input_data in iterator]
        return raw_predictions
