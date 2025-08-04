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
    import io

    # Initialize settings and classifier
    settings = Settings()
    classifier = MosquitoClassifier(settings, load_model=True)

    # Create a dummy image and in-memory byte stream
    image_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    success, encoded_image = cv2.imencode(".png", image_array)
    image_bytes = encoded_image.tobytes()
    image_stream = io.BytesIO(image_bytes)


    # Get predictions from bytes
    predictions_from_bytes = classifier.predict(image_bytes)
    print("Top prediction from bytes: ",
    f"{predictions_from_bytes[0][0]} with confidence {predictions_from_bytes[0][1]:.4f}")

    # Get predictions from stream
    predictions_from_stream = classifier.predict(image_stream)
    print("Top prediction from stream: ",
    f"{predictions_from_stream[0][0]} with confidence {predictions_from_stream[0][1]:.4f}")


    # Clean up (if not using a context manager)
    classifier.unload_model()
"""

from __future__ import annotations

import pathlib
import platform
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias, Union
from collections.abc import Sequence
import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fastai.callback.progress import ProgressCallback
from fastai.vision.all import load_learner
from fastprogress.fastprogress import progress_bar
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.core.utils import str_to_bgr
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

ClassificationPredictionType: TypeAlias = list[tuple[str, float]]
ClassificationGroundTruthType: TypeAlias = str
ImageInput = Union[np.ndarray, str, Path, Image.Image, bytes, io.BytesIO]


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
    BasePredictor[ImageInput, ClassificationPredictionType, ClassificationGroundTruthType],
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

        weights_manager = ModelWeightsManager(
            settings=settings,
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
        self.labels_map: dict[
            str,
            str,
        ] = self.settings.species_config.class_to_full_name_map
        self.num_classes: int = len(self.species_map)

    # --------------------------------------------------------------------------
    # Public Methods
    # --------------------------------------------------------------------------

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
        input_data: ImageInput,
        **kwargs: Any,
    ) -> ClassificationPredictionType:
        """Classifies the mosquito species in a single image.

        Args:
            input_data: Input image in one of the following formats:
                - np.ndarray: Image array with shape (H, W, 3) in RGB format.
                  Values can be uint8 [0, 255] or float32/float64 [0, 1].
                - str or pathlib.Path: Path to an image file.
                - PIL.Image.Image: PIL Image object.
                - bytes: In-memory bytes of an image.
                - io.BytesIO: A binary stream of an image.
            **kwargs (Any): Additional arguments (not used).

        Returns:
            A list of (species_name, confidence) tuples, sorted in
            descending order of confidence.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError: If the input data has an invalid format.
            FileNotFoundError: If the image file path doesn't exist.
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() or use a context manager.",
            )

        image = self._load_and_validate_image(input_data)

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
        input_data_batch: Sequence[ImageInput],
        show_progress: bool = False,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> list[ClassificationPredictionType]:
        """Classifies mosquito species in a batch of images.

        This method uses FastAI's test_dl for efficient batch processing and
        the `ProgressCallback` system to display progress.

        Args:
            input_data_batch: A sequence of input images.
            batch_size: Number of images per batch. Defaults to 32.
            show_progress: If True, a progress bar is displayed. Defaults to False.
            **kwargs: Additional arguments (not currently used).

        Returns:
            A list of prediction results for each image.
        """
        if not self.model_loaded:
            raise RuntimeError(
                "Model is not loaded. Call load_model() or use a context manager.",
            )
        if not input_data_batch:
            return []

        results: list[ClassificationPredictionType] = [[] for _ in input_data_batch]
        valid_images, valid_indices = self._prepare_batch_images(input_data_batch)

        if not valid_images:
            self._logger.warning("No valid images found in the batch to process.")
            return results
        try:
            with set_posix_windows():
                test_dl = self.learner.dls.test_dl(
                    valid_images,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                )
                cbs = [ProgressCallback()] if show_progress else None
                probabilities, _ = self.learner.get_preds(dl=test_dl, cbs=cbs)

                for pred_idx, original_idx in enumerate(valid_indices):
                    probs = probabilities[pred_idx]
                    species_probs = []
                    for class_idx, prob in enumerate(probs):
                        species_name = self.species_map.get(
                            class_idx,
                            f"unknown_{class_idx}",
                        )
                        species_probs.append((species_name, float(prob)))

                    species_probs.sort(key=lambda x: x[1], reverse=True)
                    results[original_idx] = species_probs

        except Exception as e:
            self._logger.error(
                f"Error during FastAI batch processing: {e}",
                exc_info=True,
            )
            self._logger.info("Falling back to individual predictions...")

            fallback_iterable = zip(valid_images, valid_indices)
            if show_progress:
                fallback_iterable = progress_bar(list(fallback_iterable), total=len(valid_images))
            for image, original_idx in fallback_iterable:
                try:
                    prediction = self.predict(image)
                    results[original_idx] = prediction
                except Exception as individual_error:
                    self._logger.error(
                        f"Failed to process image at original index {original_idx}: {individual_error}",
                    )
                    results[original_idx] = []
        return results

    def visualize(
        self,
        input_data: ImageInput,
        predictions: ClassificationPredictionType,
        save_path: str | Path | None = None,
    ) -> np.ndarray:
        """Creates a composite image with results and the input image.

        This method generates a visualization by placing the top-k predictions
        in a separate panel to the left of the image.

        Args:
            input_data: The input image (NumPy array, path, or PIL Image).
            predictions: The prediction output from the `predict` method.
            save_path: If provided, the image is saved to this path.

        Returns:
            A new image array containing the text panel and original image.

        Raises:
            ValueError: If the input data is invalid or predictions are empty.
            FileNotFoundError: If the image file path doesn't exist.
        """
        image_pil = self._load_and_validate_image(input_data)
        image_np_rgb = np.array(image_pil)

        if not predictions:
            raise ValueError("Predictions list cannot be empty")

        vis_config = self.config.visualization
        font_scale = vis_config.font_scale
        thickness = vis_config.text_thickness if vis_config.text_thickness is not None else 1
        text_color_bgr = str_to_bgr(vis_config.text_color)
        top_k = self.config.params.get("top_k", 5)
        font = cv2.FONT_HERSHEY_SIMPLEX

        img_h, img_w, _ = image_np_rgb.shape
        text_panel_width = 350
        padding = 20
        canvas_h = img_h
        canvas_w = text_panel_width + img_w
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

        y_offset = 40
        line_height = int(font_scale * 40)
        for species, conf in predictions[:top_k]:
            display_name = self.labels_map.get(species, species)
            text = f"{display_name}: {conf:.3f}"
            cv2.putText(
                canvas,
                text,
                (padding, y_offset),
                font,
                font_scale,
                text_color_bgr,
                thickness,
                lineType=cv2.LINE_AA,
            )
            y_offset += line_height

        canvas[:, text_panel_width:] = image_np_rgb

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_img_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), save_img_bgr)

        return canvas

    def visualize_report(
        self,
        report_data: dict[str, Any],
        save_path: str | Path | None = None,
    ) -> None:
        """Generates a visualization of the evaluation report.

        This function creates a figure with a text summary of key performance
        metrics and a heatmap of the confusion matrix.

        Args:
            report_data: The evaluation report from the `evaluate` method.
            save_path: If provided, the figure is saved to this path.

        Raises:
            ValueError: If `report_data` is missing required keys.
        """
        required_keys = [
            "accuracy_mean",
            "confidence_mean",
            "top_5_correct_mean",
            "count",
            "confusion_matrix",
        ]
        if not all(key in report_data for key in required_keys):
            raise ValueError("report_data is missing one or more required keys.")

        conf_matrix = np.array(report_data["confusion_matrix"])
        class_labels = self.get_species_names()

        fig, (ax_text, ax_matrix) = plt.subplots(
            1,
            2,
            figsize=(20, 8),
            gridspec_kw={"width_ratios": [1, 2.5]},
        )
        fig.suptitle("Model Evaluation Report", fontsize=20, y=1.02)

        ax_text.axis("off")
        text_content = (
            f"Summary (on {report_data['count']} samples):\n\n"
            f"Mean Accuracy (Top-1): {report_data['accuracy_mean']:.3f}\n"
            f"Mean Top-5 Accuracy:   {report_data['top_5_correct_mean']:.3f}\n\n"
            f"Mean Confidence:         {report_data['confidence_mean']:.3f}\n"
        )
        if "roc_auc" in report_data:
            text_content += f"ROC-AUC Score:           {report_data['roc_auc']:.3f}\n"
        ax_text.text(
            0.0,
            0.7,
            text_content,
            ha="left",
            va="top",
            transform=ax_text.transAxes,
            fontsize=14,
            family="monospace",
        )

        im = ax_matrix.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
        tick_marks = np.arange(len(class_labels))
        ax_matrix.set_xticks(tick_marks)
        ax_matrix.set_yticks(tick_marks)
        ax_matrix.set_xticklabels(
            class_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax_matrix.set_yticklabels(class_labels, rotation=0)
        fig.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)

        threshold = conf_matrix.max() / 2.0
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                text_color = "white" if conf_matrix[i, j] > threshold else "black"
                ax_matrix.text(
                    j,
                    i,
                    f"{conf_matrix[i, j]}",
                    ha="center",
                    va="center",
                    color=text_color,
                )
        ax_matrix.set_title("Confusion Matrix", fontsize=16)
        ax_matrix.set_xlabel("Predicted Label", fontsize=12)
        ax_matrix.set_ylabel("True Label", fontsize=12)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Report visualization saved to: {save_path}")
        plt.show()

    # --------------------------------------------------------------------------
    # Private Methods
    # --------------------------------------------------------------------------

    def _evaluate_from_prediction(
        self,
        prediction: ClassificationPredictionType,
        ground_truth: ClassificationGroundTruthType,
    ) -> dict[str, float]:
        """Calculates core evaluation metrics for a single prediction."""
        if not prediction:
            return {
                "accuracy": 0.0,
                "confidence": 0.0,
                "top_1_correct": 0.0,
                "top_5_correct": 0.0,
            }
        ground_truth_species = self.labels_map.get(ground_truth, ground_truth)
        pred_species = prediction[0][0]
        confidence = prediction[0][1]
        top_1_correct = float(pred_species == ground_truth_species)
        top_5_species = [p[0] for p in prediction[:5]]
        top_5_correct = float(ground_truth_species in top_5_species)
        return {
            "accuracy": top_1_correct,
            "confidence": confidence,
            "top_1_correct": top_1_correct,
            "top_5_correct": top_5_correct,
        }

    def _finalize_evaluation_report(
        self,
        aggregated_metrics: dict[str, float],
        predictions: Sequence[ClassificationPredictionType],
        ground_truths: Sequence[ClassificationGroundTruthType],
    ) -> dict[str, Any]:
        """Calculates and adds confusion matrix and ROC-AUC to the final report."""
        species_to_idx = {v: k for k, v in self.species_map.items()}
        class_labels = list(range(self.num_classes))
        y_true_indices, y_pred_indices, y_scores = [], [], []

        for gt, pred_list in zip(ground_truths, predictions):
            gt_str = self.labels_map.get(gt, gt)
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

    def _load_and_validate_image(self, input_data: ImageInput) -> Image.Image:
        """Loads and validates an input image from various formats.

        Args:
            input_data: Image input (numpy array, file path, PIL Image, bytes, or io.BytesIO).

        Returns:
            A validated PIL Image in RGB format.

        Raises:
            ValueError: If input format is invalid or image cannot be loaded.
            FileNotFoundError: If image file path does not exist.
        """
        if isinstance(input_data, (str, Path)):
            image_path = Path(input_data)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            try:
                image = Image.open(image_path).convert("RGB")
                return image
            except Exception as e:
                raise ValueError(f"Cannot load image from {image_path}: {e}")

        elif isinstance(input_data, Image.Image):
            return input_data.convert("RGB")

        elif isinstance(input_data, np.ndarray):
            if input_data.ndim != 3 or input_data.shape[2] != 3:
                raise ValueError(
                    f"Expected 3D RGB image, got shape: {input_data.shape}",
                )
            if input_data.dtype == np.uint8:
                return Image.fromarray(input_data)
            elif input_data.dtype in [np.float32, np.float64]:
                if input_data.max() > 1.0 or input_data.min() < 0.0:
                    raise ValueError("Float images must be in range [0, 1]")
                return Image.fromarray((input_data * 255).astype(np.uint8))
            else:
                raise ValueError(f"Unsupported numpy dtype: {input_data.dtype}")

        elif isinstance(input_data, bytes):
            try:
                return Image.open(io.BytesIO(input_data)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot load image from bytes: {e}")

        elif isinstance(input_data, io.BytesIO):
            try:
                return Image.open(input_data).convert("RGB")
            except Exception as e:
                raise ValueError(f"Cannot load image from BytesIO stream: {e}")

        else:
            raise TypeError(
                f"Unsupported input type: {type(input_data)}. "
                f"Expected np.ndarray, str, pathlib.Path, PIL.Image.Image, bytes, or io.BytesIO",
            )

    def _load_model(self) -> None:
        """Loads the pre-trained FastAI learner model from disk.

        Raises:
            RuntimeError: If the model file cannot be loaded.
        """
        with set_posix_windows():
            try:
                self.learner = load_learner(self.model_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model from {self.model_path}. " f"Ensure the file is valid. Original error: {e}",
                ) from e

    def _prepare_batch_images(
        self,
        input_data_batch: Sequence[ImageInput],
    ) -> tuple[list[Image.Image], list[int]]:
        """Prepares and validates a batch of images for processing.

        Args:
            input_data_batch: A sequence of input images.

        Returns:
            A tuple of (valid_images, valid_indices) where valid_indices
            tracks the original position of each valid image.
        """
        valid_images = []
        valid_indices = []
        for idx, input_data in enumerate(input_data_batch):
            try:
                image = self._load_and_validate_image(input_data)
                valid_images.append(image)
                valid_indices.append(idx)
            except Exception as e:
                self._logger.warning(f"Skipping image at index {idx}: {e}")
        return valid_images, valid_indices
