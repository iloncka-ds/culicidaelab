"""
Module for base predictor class that all predictors inherit from.
"""

from abc import ABC, abstractmethod
import sys
from typing import Any, TypeVar, Generic
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm as tqdm_console

import logging
from contextlib import contextmanager
from .settings import Settings
from .weights_manager_protocol import WeightsManagerProtocol
from .config_models import PredictorConfig

try:
    from tqdm.notebook import tqdm as tqdm_notebook
except ImportError:
    tqdm_notebook = tqdm_console

logger = logging.getLogger(__name__)
PredictionType = TypeVar("PredictionType")
GroundTruthType = TypeVar("GroundTruthType")


class BasePredictor(Generic[PredictionType, GroundTruthType], ABC):
    """
    Abstract base class for all predictors (detector, segmenter, classifier).
    It depends on the main Settings object for configuration.
    """

    def __init__(
        self,
        settings: Settings,
        predictor_type: str,
        weights_manager: WeightsManagerProtocol,
        load_model: bool = False,
    ):
        """
        Initialize the predictor using the global Settings object.

        Args:
            settings: The main Settings object for the library.
            predictor_type: The key for this predictor in the configuration (e.g., 'classifier').
            weights_manager: The WeightsManagerProtocol
            load_model: If True, loads model immediately.

        Raises:
            ValueError: If configuration for predictor type is not found.
        """
        self.settings = settings
        self.predictor_type = predictor_type

        # Initialize weights manager and get model path
        self._weights_manager = weights_manager
        self._model_path = self._weights_manager.ensure_weights(self.predictor_type)

        # Get predictor-specific configuration as a Pydantic model
        self._config: PredictorConfig = self._get_predictor_config()

        self._model_loaded = False
        self._model = None

        self._logger = logging.getLogger(f"culicidaelab.predictor.{self.predictor_type}")

        if load_model:
            self.load_model()

    def _get_predictor_config(self) -> PredictorConfig:
        """Get configuration for this predictor type as a Pydantic model."""
        config = self.settings.get_config(f"predictors.{self.predictor_type}")
        if not isinstance(config, PredictorConfig):
            raise ValueError(f"Configuration for predictor '{self.predictor_type}' not found or is invalid.")
        return config

    @property
    def model_path(self) -> Path:
        """Get the path to the model weights."""
        return self._model_path

    @property
    def config(self) -> PredictorConfig:
        """Get the predictor configuration Pydantic model."""
        return self._config

    @property
    def model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model from the specified path.
        Must be implemented by child classes.

        This method should set self._model and handle any loading errors.
        """
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray, **kwargs) -> PredictionType:
        """
        Make predictions on the input data.

        Args:
            input_data: Input data to make predictions on
            **kwargs: Additional predictor-specific arguments.

        Returns:
            Predictions in format specific to each predictor type

        Raises:
            RuntimeError: If model is not loaded.
        """
        pass

    @abstractmethod
    def visualize(
        self,
        input_data: np.ndarray,
        predictions: PredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Visualize predictions on the input data.

        Args:
            input_data: Original input data
            predictions: Predictions made by the model
            save_path: Optional path to save the visualization

        Returns:
            Visualization as a numpy array
        """
        pass

    def evaluate(
        self,
        ground_truth: GroundTruthType,
        prediction: PredictionType | None = None,
        input_data: np.ndarray | None = None,
        **predict_kwargs,
    ) -> dict[str, float]:
        """
        Evaluate a prediction against a ground truth.

        Either `prediction` or `input_data` must be provided.
        If `prediction` is provided, it is used directly.
        If `prediction` is None, `input_data` is used to generate a new prediction.

        Args:
            ground_truth: The ground truth annotation.
            prediction: A pre-computed prediction.
            input_data: Input data to generate a prediction from, if one isn't provided.
            **predict_kwargs: Additional arguments passed to the predict method.

        Returns:
            Dictionary containing evaluation metrics for a single item.

        Raises:
            ValueError: If neither prediction nor input_data is provided.
        """
        if prediction is None:
            if input_data is not None:
                prediction = self.predict(input_data, **predict_kwargs)
            else:
                raise ValueError("Either 'prediction' or 'input_data' must be provided.")

        return self._evaluate_from_prediction(prediction=prediction, ground_truth=ground_truth)

    @abstractmethod
    def _evaluate_from_prediction(self, prediction: PredictionType, ground_truth: GroundTruthType) -> dict[str, float]:
        """
        The core metric calculation logic for metrics for a single item.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth annotation

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def load_model(self) -> None:
        """
        Load model if not already loaded.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._model_loaded:
            logger.info(f"Model for {self.predictor_type} already loaded")
            return

        try:
            logger.info(f"Loading model for {self.predictor_type} from {self._model_path}")
            self._load_model()
            self._model_loaded = True
            logger.info(f"Successfully loaded model for {self.predictor_type}")
        except Exception as e:
            logger.error(f"Failed to load model for {self.predictor_type}: {e}")
            raise RuntimeError(f"Failed to load model for {self.predictor_type}: {e}") from e

    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        """
        if self._model_loaded:
            self._model = None
            self._model_loaded = False
            logger.info(f"Unloaded model for {self.predictor_type}")

    @contextmanager
    def model_context(self):
        """
        Context manager for temporary model loading.

        Usage:
            with predictor.model_context():
                predictions = predictor.predict(data)
        """
        was_loaded = self._model_loaded
        try:
            if not was_loaded:
                self.load_model()
            yield self
        finally:
            if not was_loaded and self._model_loaded:
                self.unload_model()

    def evaluate_batch(
        self,
        ground_truth_batch: list[GroundTruthType],
        predictions_batch: list[PredictionType] | None = None,
        input_data_batch: list[np.ndarray] | None = None,
        num_workers: int = 4,
        show_progress: bool = True,
        **predict_kwargs,
    ) -> dict[str, float]:
        """
        Evaluate on a batch of items using parallel processing for metric calculation.

        Either `predictions_batch` or `input_data_batch` must be provided.

        Args:
            ground_truth_batch: List of corresponding ground truth annotations.
            predictions_batch: A pre-computed list of predictions.
            input_data_batch: List of input data to generate predictions from.
            num_workers: Number of parallel workers for calculating metrics.
            show_progress: Whether to show progress bars.
            **predict_kwargs: Additional arguments passed to predict_batch.

        Returns:
            Dictionary containing aggregated evaluation metrics.
        """
        # Validate inputs
        if predictions_batch is None:
            if input_data_batch is not None:
                predictions_batch = self.predict_batch(
                    input_data_batch,
                    show_progress=show_progress,
                    **predict_kwargs,
                )
            else:
                raise ValueError("Either 'predictions_batch' or 'input_data_batch' must be provided.")

        if len(predictions_batch) != len(ground_truth_batch):
            raise ValueError(
                f"Number of predictions ({len(predictions_batch)}) must match "
                f"number of ground truths ({len(ground_truth_batch)}).",
            )

        # Calculate metrics for each item
        per_item_metrics = self._calculate_metrics_parallel(
            predictions_batch,
            ground_truth_batch,
            num_workers,
            show_progress,
        )

        # Aggregate results
        aggregated_metrics = self._aggregate_metrics(per_item_metrics)

        # Finalize report (e.g., add confusion matrix)
        final_report = self._finalize_evaluation_report(
            aggregated_metrics,
            predictions_batch,
            ground_truth_batch,
        )

        return final_report

    def predict_batch(
        self,
        input_data_batch: list[np.ndarray],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[PredictionType]:
        """
        Make predictions on a batch of inputs.

        This base implementation processes items serially. Subclasses with native
        batching capabilities SHOULD override this method.

        Args:
            input_data_batch: List of input data to make predictions on.
            show_progress: Whether to show a progress bar.
            **kwargs: Additional arguments passed to each predict call.

        Returns:
            List of predictions.

        Raises:
            RuntimeError: If model fails to load or predict.
        """
        if not input_data_batch:
            return []

        if not self._model_loaded:
            self.load_model()
            if not self._model_loaded:
                raise RuntimeError("Failed to load model for batch prediction")

        in_notebook = "ipykernel" in sys.modules
        tqdm_iterator = tqdm_notebook if in_notebook else tqdm_console
        iterator = input_data_batch

        if show_progress:
            iterator = tqdm_iterator(
                input_data_batch,
                desc=f"Predicting batch ({self.predictor_type})",
                leave=False,
            )

        try:
            return [self.predict(item, **kwargs) for item in iterator]
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch prediction failed: {e}") from e

    def _calculate_metrics_parallel(
        self,
        predictions: list[PredictionType],
        ground_truths: list[GroundTruthType],
        num_workers: int = 4,
        show_progress: bool = True,
    ) -> list[dict[str, float]]:
        """Calculate metrics for individual items in parallel."""
        per_item_metrics = []

        in_notebook = "ipykernel" in sys.modules
        tqdm_iterator = tqdm_notebook if in_notebook else tqdm_console

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._evaluate_from_prediction, predictions[i], ground_truths[i]): i
                for i in range(len(predictions))
            }

            iterator = as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm_iterator(iterator, total=len(future_to_idx), desc="Calculating metrics")

            for future in iterator:
                try:
                    per_item_metrics.append(future.result())
                except Exception as e:
                    logger.error(f"Error calculating metrics for item {future_to_idx[future]}: {e}")
                    per_item_metrics.append({})

        return per_item_metrics

    def _aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics from multiple evaluations."""
        if not metrics_list:
            return {}

        valid_metrics = [m for m in metrics_list if m]
        if not valid_metrics:
            logger.warning("No valid metrics found for aggregation")
            return {}

        all_keys = {k for m in valid_metrics for k in m.keys()}
        aggregated = {}
        for key in all_keys:
            values = [m[key] for m in valid_metrics if key in m]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        aggregated["count"] = len(valid_metrics)
        return aggregated

    def _finalize_evaluation_report(
        self,
        aggregated_metrics: dict[str, float],
        predictions: list[PredictionType],
        ground_truths: list[GroundTruthType],
    ) -> dict[str, float]:
        """Optional hook to post-process the final evaluation report."""
        return aggregated_metrics

    def __call__(self, input_data: np.ndarray, **kwargs) -> Any:
        """Convenience method that calls predict()."""
        if not self._model_loaded:
            self.load_model()
        return self.predict(input_data, **kwargs)

    def __enter__(self):
        """Context manager entry."""
        if not self._model_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "predictor_type": self.predictor_type,
            "model_path": str(self._model_path),
            "model_loaded": self._model_loaded,
            "config": self.config.model_dump(),
        }
