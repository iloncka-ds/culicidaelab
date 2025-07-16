"""Base predictor class that all predictors inherit from.

This module defines the `BasePredictor` abstract base class, which establishes
a common interface and functionality for all predictors in the library, such as
detectors, segmenters, and classifiers.
"""

import logging
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from tqdm import tqdm as tqdm_console

from culicidaelab.core.config_models import PredictorConfig
from culicidaelab.core.settings import Settings
from culicidaelab.core.weights_manager_protocol import WeightsManagerProtocol

try:
    from tqdm.notebook import tqdm as tqdm_notebook
except ImportError:
    tqdm_notebook = tqdm_console

logger = logging.getLogger(__name__)
PredictionType = TypeVar("PredictionType")
GroundTruthType = TypeVar("GroundTruthType")


class BasePredictor(Generic[PredictionType, GroundTruthType], ABC):
    """Abstract base class for all predictors.

    This class defines the common interface for all predictors (e.g., detector,
    segmenter, classifier). It relies on the main Settings object for
    configuration and a WeightsManager for model file management.

    Args:
        settings (Settings): The main Settings object for the library.
        predictor_type (str): The key for this predictor in the configuration
            (e.g., 'classifier').
        weights_manager (WeightsManagerProtocol): An object conforming to the
            WeightsManagerProtocol.
        load_model (bool): If True, loads the model immediately upon initialization.

    Attributes:
        settings (Settings): The main settings object.
        predictor_type (str): The type of the predictor.
        weights_manager (WeightsManagerProtocol): The manager responsible for
            providing model weights.
    """

    def __init__(
        self,
        settings: Settings,
        predictor_type: str,
        weights_manager: WeightsManagerProtocol,
        load_model: bool = False,
    ):
        """Initializes the predictor.

        Raises:
            ValueError: If the configuration for the specified `predictor_type`
                is not found in the settings.
        """
        self.settings = settings
        self.predictor_type = predictor_type

        self._weights_manager = weights_manager
        self._model_path = self._weights_manager.ensure_weights(self.predictor_type)
        self._config: PredictorConfig = self._get_predictor_config()

        self._model = None
        self._model_loaded = False
        self._logger = logging.getLogger(
            f"culicidaelab.predictor.{self.predictor_type}",
        )

        if load_model:
            self.load_model()

    def __call__(self, input_data: np.ndarray, **kwargs: Any) -> Any:
        """Convenience method that calls `predict()`."""
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

    @property
    def config(self) -> PredictorConfig:
        """Get the predictor configuration Pydantic model."""
        return self._config

    @property
    def model_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded

    @property
    def model_path(self) -> Path:
        """Gets the path to the model weights file."""
        return self._model_path

    @contextmanager
    def model_context(self):
        """A context manager for temporary model loading.

        Ensures the model is loaded upon entering the context and unloaded
        upon exiting. This is useful for managing memory in pipelines.

        Yields:
            BasePredictor: The predictor instance itself.

        Example:
            >>> with predictor.model_context():
            ...     predictions = predictor.predict(data)
        """
        was_loaded = self._model_loaded
        try:
            if not was_loaded:
                self.load_model()
            yield self
        finally:
            if not was_loaded and self._model_loaded:
                self.unload_model()

    def evaluate(
        self,
        ground_truth: GroundTruthType,
        prediction: PredictionType | None = None,
        input_data: np.ndarray | None = None,
        **predict_kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate a prediction against a ground truth.

        Either `prediction` or `input_data` must be provided. If `prediction`
        is provided, it is used directly. If `prediction` is None, `input_data`
        is used to generate a new prediction.

        Args:
            ground_truth (GroundTruthType): The ground truth annotation.
            prediction (PredictionType, optional): A pre-computed prediction.
            input_data (np.ndarray, optional): Input data to generate a
                prediction from, if one isn't provided.
            **predict_kwargs (Any): Additional arguments passed to the `predict`
                method.

        Returns:
            dict[str, float]: Dictionary containing evaluation metrics for a
            single item.

        Raises:
            ValueError: If neither `prediction` nor `input_data` is provided.
        """
        if prediction is None:
            if input_data is not None:
                prediction = self.predict(input_data, **predict_kwargs)
            else:
                raise ValueError(
                    "Either 'prediction' or 'input_data' must be provided.",
                )
        return self._evaluate_from_prediction(
            prediction=prediction,
            ground_truth=ground_truth,
        )

    def evaluate_batch(
        self,
        ground_truth_batch: list[GroundTruthType],
        predictions_batch: list[PredictionType] | None = None,
        input_data_batch: list[np.ndarray] | None = None,
        num_workers: int = 4,
        show_progress: bool = True,
        **predict_kwargs,
    ) -> dict[str, float]:
        """Evaluate on a batch of items using parallel processing.

        Either `predictions_batch` or `input_data_batch` must be provided.

        Args:
            ground_truth_batch (list[GroundTruthType]): List of corresponding
                ground truth annotations.
            predictions_batch (list[PredictionType], optional): A pre-computed
                list of predictions.
            input_data_batch (list[np.ndarray], optional): List of input data
                to generate predictions from.
            num_workers (int): Number of parallel workers for calculating metrics.
            show_progress (bool): Whether to show a progress bar.
            **predict_kwargs: Additional arguments passed to `predict_batch`.

        Returns:
            dict[str, float]: Dictionary containing aggregated evaluation metrics.

        Raises:
            ValueError: If the number of predictions does not match the number
                of ground truths.
        """
        if predictions_batch is None:
            if input_data_batch is not None:
                predictions_batch = self.predict_batch(
                    input_data_batch,
                    show_progress=show_progress,
                    **predict_kwargs,
                )
            else:
                raise ValueError(
                    "Either 'predictions_batch' or 'input_data_batch' must be provided.",
                )

        if len(predictions_batch) != len(ground_truth_batch):
            raise ValueError(
                f"Number of predictions ({len(predictions_batch)}) must match "
                f"number of ground truths ({len(ground_truth_batch)}).",
            )

        per_item_metrics = self._calculate_metrics_parallel(
            predictions_batch,
            ground_truth_batch,
            num_workers,
            show_progress,
        )
        aggregated_metrics = self._aggregate_metrics(per_item_metrics)
        final_report = self._finalize_evaluation_report(
            aggregated_metrics,
            predictions_batch,
            ground_truth_batch,
        )
        return final_report

    def get_model_info(self) -> dict[str, Any]:
        """Gets information about the loaded model.

        Returns:
            dict[str, Any]: A dictionary containing details about the model, such
            as architecture, path, etc.
        """
        return {
            "predictor_type": self.predictor_type,
            "model_path": str(self._model_path),
            "model_loaded": self._model_loaded,
            "config": self.config.model_dump(),
        }

    def load_model(self) -> None:
        """Loads the model if it is not already loaded.

        This is a convenience wrapper around `_load_model` that prevents
        reloading.

        Raises:
            RuntimeError: If model loading fails.
        """
        if self._model_loaded:
            self._logger.info(f"Model for {self.predictor_type} already loaded")
            return

        try:
            self._logger.info(
                f"Loading model for {self.predictor_type} from {self._model_path}",
            )
            self._load_model()
            self._model_loaded = True
            self._logger.info(f"Successfully loaded model for {self.predictor_type}")
        except Exception as e:
            self._logger.error(f"Failed to load model for {self.predictor_type}: {e}")
            raise RuntimeError(
                f"Failed to load model for {self.predictor_type}: {e}",
            ) from e

    def predict_batch(
        self,
        input_data_batch: list[np.ndarray],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[PredictionType]:
        """Makes predictions on a batch of inputs.

        This base implementation processes items serially. Subclasses with
        native batching capabilities SHOULD override this method.

        Args:
            input_data_batch (list[np.ndarray]): List of input data to make
                predictions on.
            show_progress (bool): Whether to show a progress bar.
            **kwargs (Any): Additional arguments passed to each `predict` call.

        Returns:
            list[PredictionType]: List of predictions.

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
            self._logger.error(f"Batch prediction failed: {e}", exc_info=True)
            raise RuntimeError(f"Batch prediction failed: {e}") from e

    def unload_model(self) -> None:
        """Unloads the model to free memory."""
        if self._model_loaded:
            self._model = None
            self._model_loaded = False
            self._logger.info(f"Unloaded model for {self.predictor_type}")

    # Abstract Methods
    @abstractmethod
    def _evaluate_from_prediction(
        self,
        prediction: PredictionType,
        ground_truth: GroundTruthType,
    ) -> dict[str, float]:
        """The core metric calculation logic for a single item.

        Args:
            prediction (PredictionType): Model prediction.
            ground_truth (GroundTruthType): Ground truth annotation.

        Returns:
            dict[str, float]: Dictionary containing evaluation metrics.
        """
        pass

    @abstractmethod
    def _load_model(self) -> None:
        """Loads the model from the path specified in the configuration.

        This method must be implemented by child classes. It should handle
        the specifics of loading a model file (e.g., PyTorch, TensorFlow)
        and assign it to an internal attribute like `self._model`.

        Raises:
            RuntimeError: If the model file cannot be found or loaded.
        """
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray, **kwargs: Any) -> PredictionType:
        """Makes a prediction on a single input data sample.

        Args:
            input_data (np.ndarray): The input data (e.g., an image as a NumPy
                array) to make a prediction on.
            **kwargs (Any): Additional predictor-specific arguments.

        Returns:
            PredictionType: The prediction result, with a format specific to the
            predictor type.

        Raises:
            RuntimeError: If the model is not loaded before calling this method.
        """
        pass

    @abstractmethod
    def visualize(
        self,
        input_data: np.ndarray,
        predictions: PredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Visualizes the predictions on the input data.

        Args:
            input_data (np.ndarray): The original input data (e.g., an image).
            predictions (PredictionType): The prediction result obtained from
                the `predict` method.
            save_path (str | Path, optional): An optional path to save the
                visualization to a file.

        Returns:
            np.ndarray: A NumPy array representing the visualized image.
        """
        pass

    # Protected Methods
    def _aggregate_metrics(
        self,
        metrics_list: list[dict[str, float]],
    ) -> dict[str, float]:
        """Aggregates metrics from multiple evaluations."""
        if not metrics_list:
            return {}

        valid_metrics = [m for m in metrics_list if m]
        if not valid_metrics:
            self._logger.warning("No valid metrics found for aggregation")
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

    def _calculate_metrics_parallel(
        self,
        predictions: list[PredictionType],
        ground_truths: list[GroundTruthType],
        num_workers: int = 4,
        show_progress: bool = True,
    ) -> list[dict[str, float]]:
        """Calculates metrics for individual items in parallel."""
        per_item_metrics = []
        in_notebook = "ipykernel" in sys.modules
        tqdm_iterator = tqdm_notebook if in_notebook else tqdm_console

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._evaluate_from_prediction,
                    predictions[i],
                    ground_truths[i],
                ): i
                for i in range(len(predictions))
            }
            iterator = as_completed(future_to_idx)
            if show_progress:
                iterator = tqdm_iterator(
                    iterator,
                    total=len(future_to_idx),
                    desc="Calculating metrics",
                )
            for future in iterator:
                try:
                    per_item_metrics.append(future.result())
                except Exception as e:
                    self._logger.error(
                        f"Error calculating metrics for item {future_to_idx[future]}: {e}",
                    )
                    per_item_metrics.append({})
        return per_item_metrics

    def _finalize_evaluation_report(
        self,
        aggregated_metrics: dict[str, float],
        predictions: list[PredictionType],
        ground_truths: list[GroundTruthType],
    ) -> dict[str, float]:
        """Optional hook to post-process the final evaluation report."""
        return aggregated_metrics

    def _get_predictor_config(self) -> PredictorConfig:
        """Gets the configuration for this predictor.

        Returns:
            PredictorConfig: A Pydantic `PredictorConfig` model for this
            predictor instance.

        Raises:
            ValueError: If the configuration is invalid.
        """
        config = self.settings.get_config(f"predictors.{self.predictor_type}")
        if not isinstance(config, PredictorConfig):
            raise ValueError(
                f"Configuration for predictor '{self.predictor_type}' not found or is invalid.",
            )
        return config
