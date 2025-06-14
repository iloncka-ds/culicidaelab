"""
Module for base predictor class that all predictors inherit from.
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .settings import Settings
from .model_weights_manager import ModelWeightsManager


class BasePredictor(ABC):
    """
    Abstract base class for all predictors (detector, segmenter, classifier).
    It depends on the main Settings object for configuration.
    """

    def __init__(
        self,
        settings: Settings,
        predictor_type: str,
        load_model: bool = False,
    ):
        """
        Initialize the predictor using the global Settings object.

        Args:
            settings: The main Settings object for the library.
            predictor_type: The key for this predictor in the configuration (e.g., 'classifier').
            load_model: If True, loads model immediately.
        """
        self.settings = settings
        self.predictor_type = predictor_type
        weights_manager = ModelWeightsManager(self.settings)
        # The ensure_weights method returns the validated path.
        self.model_path = weights_manager.ensure_weights(self.predictor_type)

        # The base class now fetches the specific config block for the child.
        self.config = self.settings.get_config(f"predictors.{self.predictor_type}")
        if self.config is None:
            raise ValueError(f"Configuration for predictor '{self.predictor_type}' not found.")

        self.model_loaded = False
        if load_model:
            self.load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """
        Load the model from the specified path.
        Must be implemented by child classes.
        """
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Any:
        """
        Make predictions on the input data.

        Args:
            input_data: Input data to make predictions on

        Returns:
            Predictions in format specific to each predictor type
        """
        pass

    @abstractmethod
    def visualize(
        self,
        input_data: np.ndarray,
        predictions: Any,
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

    @abstractmethod
    def evaluate(
        self,
        input_data: np.ndarray,
        ground_truth: Any,
    ) -> dict[str, float]:
        """
        Evaluate model predictions against ground truth.

        Args:
            input_data: Input data to evaluate on
            ground_truth: Ground truth annotations

        Returns:
            Dictionary containing evaluation metrics
        """
        pass

    def evaluate_batch(
        self,
        input_data_batch: list[np.ndarray],
        ground_truth_batch: list[Any],
        num_workers: int = 4,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """
        Evaluate model on a batch of inputs using parallel processing.

        Args:
            input_data_batch: List of input data to evaluate on
            ground_truth_batch: List of corresponding ground truth annotations
            num_workers: Number of parallel workers (default: 4)
            batch_size: Size of batches to process at once (default: 32)

        Returns:
            Dictionary containing aggregated evaluation metrics
        """
        if len(input_data_batch) != len(ground_truth_batch):
            raise ValueError("Number of inputs must match number of ground truth annotations")

        if len(input_data_batch) == 0:
            raise ValueError("Input batch cannot be empty")

        def process_batch(batch_idx: int) -> dict[str, float]:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_data_batch))

            batch_metrics = []
            for i in range(start_idx, end_idx):
                metrics = self.evaluate(input_data_batch[i], ground_truth_batch[i])
                batch_metrics.append(metrics)

            return self._aggregate_metrics(batch_metrics)

        num_batches = (len(input_data_batch) + batch_size - 1) // batch_size
        batch_indices = range(num_batches)

        all_metrics = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_batch, idx) for idx in batch_indices]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating batches"):
                try:
                    batch_result = future.result()
                    all_metrics.append(batch_result)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

        return self._aggregate_metrics(all_metrics)

    def predict_batch(
        self,
        input_data_batch: list[np.ndarray],
        num_workers: int = 4,
        batch_size: int = 32,
    ) -> list[Any]:
        """
        Make predictions on a batch of inputs in parallel.

        Args:
            input_data_batch: List of input data to make predictions on
            num_workers: Number of parallel workers
            batch_size: Size of batches to process at once

        Returns:
            List of predictions
        """
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(input_data_batch), batch_size):
                batch = input_data_batch[i : i + batch_size]
                future = executor.submit(self._predict_batch, batch)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

        return results

    def _predict_batch(self, batch: list[np.ndarray]) -> list[Any]:
        """
        Process a batch of inputs.

        Args:
            batch: List of input data

        Returns:
            List of predictions for the batch
        """
        return [self.predict(x) for x in batch]

    def _aggregate_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """
        Aggregate metrics from multiple evaluations.

        Args:
            metrics_list: List of metric dictionaries to aggregate

        Returns:
            Dictionary containing aggregated metrics
        """
        if not metrics_list:
            return {}

        metric_keys = metrics_list[0].keys()

        aggregated = {}

        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        return aggregated

    def load_model(self) -> None:
        """Load model if not already loaded."""
        if not self.model_loaded:
            self._load_model()
            self.model_loaded = True

    def __call__(self, input_data: np.ndarray) -> Any:
        """
        Make predictions on input data.
        Convenience method that calls predict().

        Args:
            input_data: Input data to make predictions on

        Returns:
            Model predictions
        """
        if not self.model_loaded:
            self.load_model()
        return self.predict(input_data)
