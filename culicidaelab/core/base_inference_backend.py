import logging
from abc import ABC, abstractmethod

from typing import Any, Generic, TypeVar

from fastprogress.fastprogress import progress_bar
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol

logger = logging.getLogger(__name__)

InputDataType = TypeVar("InputDataType")
PredictionType = TypeVar("PredictionType")


class BaseInferenceBackend(Generic[InputDataType, PredictionType], ABC):
    """
    An abstract base class for inference backends that provides a default
    iterative implementation for predict_batch.
    """

    def __init__(
        self,
        weights_manager: WeightsManagerProtocol,
    ):
        self.weights_manager = weights_manager
        self.model = None

    @abstractmethod
    def load_model(self, predictor_type: str, **kwargs: Any) -> None: ...

    @abstractmethod
    def predict(self, input_data: InputDataType, **kwargs: Any) -> PredictionType:
        """Must be implemented by all subclasses."""
        ...

    def unload_model(self):
        self.model = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_batch(
        self,
        input_data_batch: list[InputDataType],
        predictor_type: str,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[PredictionType]:
        """Makes predictions on a batch of inputs by delegating to the backend."""
        if not input_data_batch:
            return []

        if not self.is_loaded:
            self.load_model(predictor_type)
        iterator = input_data_batch
        if show_progress:
            iterator = progress_bar(input_data_batch, total=len(input_data_batch))
        raw_predictions = [self.predict(input_data, **kwargs) for input_data in iterator]
        return raw_predictions
