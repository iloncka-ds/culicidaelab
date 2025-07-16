"""Module for mosquito species classification using FastAI.

This module provides the MosquitoClassifier class for identifying mosquito species
from an image. It leverages the FastAI framework and can use various model
architectures available in the `timm` library. The classifier is designed to be
initialized with project-wide settings and can be used to predict species for
single images or batches.

Example:
    from culicidaelab.core.settings import Settings
    from culicidaelab.predictors import MosquitoClassifier
    import numpy as np

    # Initialize settings and classifier
    settings = Settings()
    classifier = MosquitoClassifier(settings, load_model=True)

    # Create a dummy image
    image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

    # Get predictions
    predictions = classifier.predict(image)
    print(f"Top prediction: {predictions[0][0]} with confidence {predictions[0][1]:.4f}")

    # Clean up (if not using a context manager)
    classifier.unload_model()
"""

from __future__ import annotations

import pathlib
import platform
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
from culicidaelab.core.provider_service import ProviderService
from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

ClassificationPredictionType: TypeAlias = list[tuple[str, float]]
ClassificationGroundTruthType: TypeAlias = str


@contextmanager
def set_posix_windows():
    """Temporarily patch pathlib for Windows FastAI model loading.

    This context manager addresses a common issue where FastAI models trained
    on a POSIX-based system (like Linux or macOS) fail to load on Windows
    due to differences in path object serialization. It temporarily makes
    `pathlib.PosixPath` behave like `pathlib.WindowsPath` during model loading.

    Yields:
        None: Executes the code within the `with` block.
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


class MosquitoClassifier(
    BasePredictor[ClassificationPredictionType, ClassificationGroundTruthType],
):
    """Classifies mosquito species from an image using a FastAI model.

    This class provides methods to load a pre-trained model, predict species
    from single or batches of images, evaluate model performance, and visualize
    the classification results.

    Args:
        settings (Settings): The main settings object for the library, which
            contains configuration for paths, models, and species.
        load_model (bool, optional): If True, the model weights are loaded
            immediately upon initialization. Defaults to False.

    Attributes:
        arch (str): The model architecture (e.g., 'convnext_tiny').
        data_dir (Path): The directory where datasets are stored.
        species_map (dict[int, str]): A mapping from class indices to species names.
        num_classes (int): The total number of species classes.
        learner: The loaded FastAI learner object, available after `load_model()`.
    """

    def __init__(self, settings: Settings, load_model: bool = False) -> None:
        """Initializes the MosquitoClassifier."""
        provider_service = ProviderService(settings)
        weights_manager = ModelWeightsManager(
            settings=settings,
            provider_service=provider_service,
        )
        super().__init__(
            settings=settings,
            predictor_type="classifier",
            weights_manager=weights_manager,
            load_model=load_model,
        )

        self.arch: str | None = self.config.model_arch
        self.data_dir: Path = self.settings.dataset_dir
        self.species_map: dict[int, str] = self.settings.species_config.species_map
        self.num_classes: int = len(self.species_map)

    def get_class_index(self, species_name: str) -> int | None:
        """Retrieves the class index for a given species name.

        Args:
            species_name (str): The name of the species.

        Returns:
            int | None: The corresponding class index if found, otherwise None.
        """
        return self.settings.species_config.get_index_by_species(species_name)

    def get_species_names(self) -> list[str]:
        """Gets a sorted list of all species names known to the classifier.

        The list is ordered by the class index.

        Returns:
            list[str]: A list of species names.
        """
        return [self.species_map[i] for i in sorted(self.species_map.keys())]

    def predict(
        self,
        input_data: np.ndarray,
        **kwargs: Any,
    ) -> ClassificationPredictionType:
        """Classifies the mosquito species in a single image.

        Args:
            input_data (np.ndarray): An input image as a NumPy array with a
                shape of (H, W, 3) in RGB format. Values can be uint8
                [0, 255] or float [0, 1].
            **kwargs (Any): Additional arguments (not used).

        Returns:
            ClassificationPredictionType: A list of (species_name, confidence)
            tuples, sorted in descending order of confidence.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If the input data has an invalid shape, data type,
                or is not a valid image.
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() or use a context manager.",
            )

        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        if input_data.ndim != 3 or input_data.shape[2] != 3:
            raise ValueError(f"Expected 3D RGB image, got shape: {input_data.shape}")

        if input_data.dtype == np.uint8:
            image = Image.fromarray(input_data)
        elif input_data.dtype in [np.float32, np.float64]:
            image = Image.fromarray((input_data * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unsupported dtype: {input_data.dtype}")

        with set_posix_windows():
            _, _, probabilities = self.learner.predict(image)

        species_probs = []
        for idx, prob in enumerate(probabilities):
            species_name = self.species_map.get(idx, f"unknown_{idx}")
            species_probs.append((species_name, float(prob)))

        species_probs.sort(key=lambda x: x[1], reverse=True)
        return species_probs

    def predict_batch(
        self,
        input_data_batch: list[Any],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[ClassificationPredictionType]:
        """Classifies mosquito species in a batch of images.

        Note: This method currently iterates and calls `predict` for each image.
        True batch processing is not yet implemented.

        Args:
            input_data_batch (list[Any]): A list of input images, where each
                image is a NumPy array.
            show_progress (bool, optional): If True, a progress bar is displayed.
                Defaults to False.
            **kwargs (Any): Additional arguments passed to `predict`.

        Returns:
            list[ClassificationPredictionType]: A list of prediction results,
            where each result corresponds to an input image.
        """
        results = []
        for img in input_data_batch:
            results.append(self.predict(img, **kwargs))
        return results

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: ClassificationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Overlays classification results on an image.

        This method draws the top-k predictions and their confidence scores
        onto the input image.

        Args:
            input_data (np.ndarray): The original image (H, W, 3) as a NumPy array.
            predictions (ClassificationPredictionType): The prediction output from
                the `predict` method.
            save_path (str | Path | None, optional): If provided, the visualized
                image will be saved to this path. Defaults to None.

        Returns:
            np.ndarray: A new image array with the prediction text overlaid.

        Raises:
            ValueError: If the input data is not a 3D image or if the
                predictions list is empty.
        """
        if input_data.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape: {input_data.shape}")

        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        vis_img = input_data.copy()
        vis_config = self.config.visualization
        font_scale = vis_config.font_scale
        thickness = vis_config.text_thickness if vis_config.text_thickness is not None else 1
        color = str_to_bgr(vis_config.text_color)
        top_k = self.config.params.get("top_k", 5)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        for species, conf in predictions[:top_k]:
            text = f"{species}: {conf:.3f}"
            cv2.putText(
                vis_img,
                text,
                (10, y_offset),
                font,
                font_scale,
                color,
                thickness,
            )
            y_offset += 35

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), save_img)

        return vis_img

    def _evaluate_from_prediction(
        self,
        prediction: ClassificationPredictionType,
        ground_truth: ClassificationGroundTruthType,
    ) -> dict[str, float]:
        """Calculates core evaluation metrics for a single prediction.

        Args:
            prediction (ClassificationPredictionType): The model's prediction, which is
                a list of (species, confidence) tuples.
            ground_truth (ClassificationGroundTruthType): The true species name as a string.

        Returns:
            dict[str, float]: A dictionary of metrics including accuracy,
            confidence, top-1 correctness, and top-5 correctness.
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
        top_1_correct = float(pred_species == ground_truth)

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
        """Calculates and adds confusion matrix and ROC-AUC to the final report.

        Args:
            aggregated_metrics (dict[str, float]): A dictionary of metrics that have
                already been aggregated over the dataset.
            predictions (list[ClassificationPredictionType]): The list of all predictions.
            ground_truths (list[ClassificationGroundTruthType]): The list of all ground truths.

        Returns:
            dict[str, Any]: The updated report with the confusion matrix
            (as a list of lists) and the overall ROC-AUC score.
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

        if y_true_indices and y_pred_indices:
            valid_indices = [i for i, p_idx in enumerate(y_pred_indices) if p_idx != -1]
            if valid_indices:
                cm_y_true = [y_true_indices[i] for i in valid_indices]
                cm_y_pred = [y_pred_indices[i] for i in valid_indices]
                conf_matrix = confusion_matrix(
                    cm_y_true,
                    cm_y_pred,
                    labels=class_labels,
                )
                aggregated_metrics["confusion_matrix"] = conf_matrix.tolist()

        if y_scores and y_true_indices and len(np.unique(y_true_indices)) > 1:
            y_true_binarized = label_binarize(y_true_indices, classes=class_labels)
            try:
                roc_auc = roc_auc_score(
                    y_true_binarized,
                    np.array(y_scores),
                    multi_class="ovr",
                )
                aggregated_metrics["roc_auc"] = roc_auc
            except ValueError as e:
                self._logger.warning(f"Could not compute ROC AUC score: {e}")
                aggregated_metrics["roc_auc"] = 0.0

        return aggregated_metrics

    def _load_model(self) -> None:
        """Loads the pre-trained FastAI learner model from disk.

        This method uses the `set_posix_windows` context manager to ensure
        path compatibility across operating systems.

        Raises:
            RuntimeError: If the model file cannot be loaded, either because
                it is missing, corrupted, or dependencies are not met.
        """
        with set_posix_windows():
            try:
                self.learner = load_learner(self.model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load existing model from {self.model_path}. "
                    f"Ensure the model file is valid and all dependencies are installed. Original error: {e}",
                ) from e
