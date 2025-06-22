"""Module for mosquito species classification using FastAI and timm.

This module provides a classifier for mosquito species identification using
pre-trained deep learning models with FastAI framework and timm backbones.
"""

from __future__ import annotations

import platform
import pathlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from fastai.vision.all import load_learner
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr

ClassificationPredictionType: TypeAlias = list[tuple[str, float]]
ClassificationGroundTruthType: TypeAlias = str


@contextmanager
def set_posix_windows():
    """Context manager to handle path compatibility between Windows and POSIX systems.

    On Windows systems, temporarily replaces PosixPath with WindowsPath to ensure
    compatibility with FastAI models that may have been trained on POSIX systems.

    Yields:
        None
    """
    if platform.system() == "Windows":
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
    else:
        yield


class MosquitoClassifier(BasePredictor[ClassificationPredictionType, ClassificationGroundTruthType]):
    """Mosquito species classifier using FastAI and timm models."""

    def __init__(
        self,
        settings: Settings,
        load_model: bool = False,
    ) -> None:
        """
        Initialize the mosquito classifier.

        Args:
            settings: The main Settings object for the library.
            load_model: Whether to load the model immediately.
        """
        super().__init__(settings=settings, predictor_type="classifier", load_model=load_model)

        # These attributes are initialized after super().__init__ which sets up self.config
        self.arch = self.config.model_arch
        self.data_dir = self.settings.dataset_dir
        self.species_map = self.settings.species_config.species_map
        self.num_classes = len(self.species_map)

    def _load_model(self) -> None:
        """Load the FastAI model with timm backbone.

        Attempts to load a pre-trained FastAI learner. If loading fails,
        creates a new learner with the specified architecture.

        Raises:
            Exception: If model loading fails and a new learner cannot be created.
        """
        with set_posix_windows():
            try:
                self.learner = load_learner(self.model_path)
            except Exception as e:
                # Suppress the original error and raise a more informative one
                raise RuntimeError(
                    f"Failed to load existing model from {self.model_path}. "
                    f"Ensure the model file is valid and all dependencies are installed. Original error: {e}",
                ) from e

    def predict(self, input_data: np.ndarray, **kwargs: Any) -> ClassificationPredictionType:
        """Classify mosquito species in an image.

        Args:
            input_data: Input image as numpy array with shape (H, W, C) where C=3 for RGB.
                Values should be in range [0, 255] for uint8 or [0, 1] for float.

        Returns:
            A list of (species_name, confidence_score) tuples for all classes,
            ordered by confidence in descending order.

        Raises:
            ValueError: If input_data has invalid shape or dtype.
            RuntimeError: If model is not loaded.
        """
        if not self.model_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() or use a context manager.")

        # Convert input to numpy array if it isn't already
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Validate input
        if input_data.ndim != 3 or input_data.shape[2] != 3:
            raise ValueError(f"Expected 3D RGB image, got shape: {input_data.shape}")

        # Convert numpy array to PIL Image
        if input_data.dtype == np.uint8:
            image = Image.fromarray(input_data)
        elif input_data.dtype in [np.float32, np.float64]:
            image = Image.fromarray((input_data * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unsupported dtype: {input_data.dtype}")

        # Get predictions
        with set_posix_windows():
            _, _, probabilities = self.learner.predict(image)

        # Convert predictions to list of (species, confidence) tuples
        species_probs = []
        for idx, prob in enumerate(probabilities):
            species_name = self.species_map.get(idx, f"unknown_{idx}")
            species_probs.append((species_name, float(prob)))

        # Sort by confidence (descending)
        species_probs.sort(key=lambda x: x[1], reverse=True)

        return species_probs

    def predict_batch(
        self,
        input_data_batch: list[Any],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[ClassificationPredictionType]:
        """Classify mosquito species in a batch of images.

        Args:
            input_data_batch: List of input images as numpy arrays.
            show_progress: Whether to show a progress bar.
            **kwargs: Additional keyword arguments.

        Returns:
            List of predictions for each input image.
        """
        results = []
        for img in input_data_batch:
            results.append(self.predict(img, **kwargs))
        return results

    def _evaluate_from_prediction(
        self,
        prediction: ClassificationPredictionType,
        ground_truth: ClassificationGroundTruthType,
    ) -> dict[str, float]:
        """
        The core metric calculation logic for a single item.

        Args:
            prediction: Model prediction (list of (species, confidence) tuples).
            ground_truth: Ground truth species name.

        Returns:
            Dictionary containing evaluation metrics for a single item.
        """
        if not prediction:
            return {
                "accuracy": 0.0,
                "confidence": 0.0,
                "top_1_correct": 0.0,
                "top_5_correct": 0.0,
            }

        pred_species = prediction[0][0]
        confidence = prediction[0][1]

        # Check if prediction is correct
        top_1_correct = float(pred_species == ground_truth)

        # Check if ground truth is in top 5 predictions
        top_5_species = [p[0] for p in prediction[:5]]
        top_5_correct = float(ground_truth in top_5_species)

        return {
            "accuracy": top_1_correct,
            "confidence": confidence,
            "top_1_correct": top_1_correct,
            "top_5_correct": top_5_correct,
        }

    def _finalize_evaluation_report(
        self,
        aggregated_metrics: dict[str, float],
        predictions: list[ClassificationPredictionType],
        ground_truths: list[ClassificationGroundTruthType],
    ) -> dict[str, Any]:
        """
        Post-process the final report to add confusion matrix and ROC-AUC score.

        Args:
            aggregated_metrics: Aggregated metrics from individual evaluations.
            predictions: All predictions (list of lists of (species, conf) tuples).
            ground_truths: All ground truth annotations (list of species names).

        Returns:
            Final evaluation report including the confusion matrix and ROC AUC.
        """
        species_to_idx = {v: k for k, v in self.species_map.items()}
        class_labels = list(range(self.num_classes))

        y_true_indices, y_pred_indices, y_scores = [], [], []

        for gt_str, pred_list in zip(ground_truths, predictions):
            if gt_str in species_to_idx and pred_list:
                true_idx = species_to_idx[gt_str]
                pred_str = pred_list[0][0]
                pred_idx = species_to_idx.get(pred_str, -1)

                y_true_indices.append(true_idx)
                y_pred_indices.append(pred_idx)

                prob_vector = [0.0] * self.num_classes
                for species, conf in pred_list:
                    class_idx = species_to_idx.get(species)
                    if class_idx is not None:
                        prob_vector[class_idx] = conf
                y_scores.append(prob_vector)

        # Calculate Confusion Matrix
        if y_true_indices and y_pred_indices:
            valid_indices = [i for i, p_idx in enumerate(y_pred_indices) if p_idx != -1]
            if valid_indices:
                cm_y_true = [y_true_indices[i] for i in valid_indices]
                cm_y_pred = [y_pred_indices[i] for i in valid_indices]
                conf_matrix = confusion_matrix(cm_y_true, cm_y_pred, labels=class_labels)
                aggregated_metrics["confusion_matrix"] = conf_matrix.tolist()

        # Calculate ROC AUC Score
        if y_scores and y_true_indices and len(np.unique(y_true_indices)) > 1:
            y_true_binarized = label_binarize(y_true_indices, classes=class_labels)
            try:
                roc_auc = roc_auc_score(y_true_binarized, np.array(y_scores), multi_class="ovr")
                aggregated_metrics["roc_auc"] = roc_auc
            except ValueError as e:
                self._logger.warning(f"Could not compute ROC AUC score: {e}")
                aggregated_metrics["roc_auc"] = 0.0

        return aggregated_metrics

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: ClassificationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Visualize classification predictions on the image.

        Args:
            input_data: Original image as numpy array with shape (H, W, C).
            predictions: List of (species_name, confidence_score) tuples from predict().
            save_path: Optional path to save the visualization image.

        Returns:
            Image with prediction text overlaid as numpy array with same shape as input.

        Raises:
            ValueError: If input_data or predictions have invalid format.
        """
        if input_data.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape: {input_data.shape}")

        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        vis_img = input_data.copy()

        vis_config = self.config.visualization
        font_scale = vis_config.font_scale
        thickness = vis_config.thickness
        color = str_to_bgr(vis_config.text_color)
        top_k = self.config.params.get("top_k", 5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        for species, conf in predictions[:top_k]:
            text = f"{species}: {conf:.3f}"
            cv2.putText(vis_img, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 35

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), save_img)

        return vis_img

    def get_species_names(self) -> list[str]:
        """Get list of all species names in order of class indices."""
        return [self.species_map[i] for i in sorted(self.species_map.keys())]

    def get_class_index(self, species_name: str) -> int | None:
        """Get class index for a given species name."""
        return self.settings.species_config.get_index_by_species(species_name)
