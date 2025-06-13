"""Module for mosquito species classification using FastAI and timm.

This module provides a classifier for mosquito species identification using
pre-trained deep learning models with FastAI framework and timm backbones.
"""

from __future__ import annotations

import platform
import pathlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import timm
import torch
from fastai.vision.all import (
    CategoryBlock,
    CrossEntropyLossFlat,
    DataBlock,
    ImageBlock,
    RandomSplitter,
    Resize,
    aug_transforms,
    get_image_files,
    load_learner,
    parent_label,
    vision_learner,
)
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings


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


class MosquitoClassifier(BasePredictor):
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
        # CHANGED: This now calls the new BasePredictor.__init__
        # It passes the settings object and identifies itself as the "classifier".
        super().__init__(settings=settings, predictor_type="classifier", load_model=load_model)

        self.arch = self.config.model_arch
        self.data_dir = Path(self.config.data.data_dir) if hasattr(self.config.data, "data_dir") else None
        self.species_map = self.settings.species_config.species_map
        self.num_classes = len(self.species_map)

        print(f"Initialized classifier with {self.num_classes} species classes")
        print(f"Using architecture: {self.arch}")


    def _load_model(self) -> None:
        """Load the FastAI model with timm backbone.

        Attempts to load a pre-trained FastAI learner. If loading fails,
        creates a new learner with the specified architecture.

        Raises:
            Exception: If model loading fails and a new learner cannot be created.
        """
        with set_posix_windows():
            try:
                print(f"Loading model from: {self.model_path}")
                self.learner = load_learner(self.model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Failed to load existing model: {e}")
                print("Creating new learner...")
                self.learner = self._create_cls_learner(str(self.model_path))

    def predict(self, input_data: np.ndarray) -> list[tuple[str, float]]:
        """Classify mosquito species in an image.

        Args:
            input_data: Input image as numpy array with shape (H, W, C) where C=3 for RGB.
                Values should be in range [0, 255] for uint8 or [0, 1] for float.

        Returns:
            List of tuples containing (species_name, confidence_score) ordered by
            confidence in descending order. Length determined by config.model.top_k.

        Raises:
            ValueError: If input_data has invalid shape or dtype.
        """
        if not self.model_loaded:
            self.load_model()

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
        pred_class, pred_idx, probabilities = self.learner.predict(image)

        # Convert predictions to list of (species, confidence) tuples
        species_probs = []
        for idx, prob in enumerate(probabilities):
            species_name = self.species_map.get(idx, f"unknown_{idx}")
            species_probs.append((species_name, float(prob)))

        # Sort by confidence (descending)
        species_probs.sort(key=lambda x: x[1], reverse=True)

        # Return top predictions based on config
        top_k = getattr(self.config.model, "top_k", 5)
        return species_probs[:top_k]

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: list[tuple[str, float]],
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

        # Create visualization copy
        vis_img = input_data.copy()

        # Get visualization parameters from config
        vis_config = getattr(self.config, "visualization", None)
        font_scale = getattr(vis_config, "font_scale", 0.7) if vis_config else 0.7
        thickness = getattr(vis_config, "text_thickness", 2) if vis_config else 2
        color = getattr(vis_config, "text_color", (0, 255, 0)) if vis_config else (0, 255, 0)

        # Add text for each prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        for species, conf in predictions:
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

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert RGB to BGR for OpenCV saving
            save_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_path), save_img)

        return vis_img

    def evaluate(
        self,
        input_data: np.ndarray,
        ground_truth: str,
    ) -> dict[str, float]:
        """Evaluate classification predictions against ground truth.

        Args:
            input_data: Input image as numpy array.
            ground_truth: Ground truth species name as string.

        Returns:
            Dictionary containing evaluation metrics:
                - accuracy: 1.0 if top prediction matches ground truth, else 0.0
                - confidence: Confidence score of top prediction
                - top_1_correct: Same as accuracy (for consistency)
                - top_5_correct: 1.0 if ground truth is in top 5 predictions

        Raises:
            ValueError: If ground_truth is not a valid species name.
        """
        predictions = self.predict(input_data)

        if not predictions:
            return {
                "accuracy": 0.0,
                "confidence": 0.0,
                "top_1_correct": 0.0,
                "top_5_correct": 0.0,
            }

        pred_species = predictions[0][0]
        confidence = predictions[0][1]

        # Check if prediction is correct
        top_1_correct = float(pred_species == ground_truth)

        # Check if ground truth is in top 5 predictions
        top_5_species = [p[0] for p in predictions[:5]]
        top_5_correct = float(ground_truth in top_5_species)

        return {
            "accuracy": top_1_correct,
            "confidence": confidence,
            "top_1_correct": top_1_correct,
            "top_5_correct": top_5_correct,
        }

    def evaluate_batch(
        self,
        input_data_batch: list[np.ndarray],
        ground_truth_batch: list[str],
        num_workers: int = 4,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Evaluate model on a batch of inputs with optimized processing.

        This method processes predictions in batches for efficiency and calculates
        comprehensive evaluation metrics including confusion matrix and ROC-AUC.

        Args:
            input_data_batch: List of input images as numpy arrays.
            ground_truth_batch: List of corresponding ground truth species names.
            num_workers: Number of parallel workers for processing. Defaults to 4.
            batch_size: Size of batches to process at once. Defaults to 32.

        Returns:
            Dictionary containing aggregated evaluation metrics:
                - accuracy, precision, recall, f1: Classification metrics
                - top_1, top_5: Top-k accuracy metrics
                - roc_auc: Area under ROC curve for multi-class classification
                - confusion_matrix: Confusion matrix as nested list
                - *_std: Standard deviation for each metric across batches

        Raises:
            ValueError: If input and ground truth batch sizes don't match or are empty.
        """
        if len(input_data_batch) != len(ground_truth_batch):
            raise ValueError(
                f"Batch size mismatch: {len(input_data_batch)} inputs vs "
                f"{len(ground_truth_batch)} ground truth labels",
            )

        if len(input_data_batch) == 0:
            raise ValueError("Input batch cannot be empty")

        if not self.model_loaded:
            self.load_model()

        print("Converting images to PIL format...")
        images = []
        for img in tqdm(input_data_batch, desc="Converting images"):
            if img.dtype == np.uint8:
                images.append(Image.fromarray(img))
            else:
                images.append(Image.fromarray((img * 255).astype(np.uint8)))

        print("Getting model predictions...")
        all_predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="Processing batches"):
                batch = images[i : i + batch_size]
                batch_preds = []

                for img in batch:
                    pred_class, pred_idx, probs = self.learner.predict(img)
                    batch_preds.append((pred_idx.item(), probs))

                all_predictions.extend(batch_preds)

        species_to_idx = {v: k for k, v in self.species_map.items()}

        def process_evaluation_batch(batch_idx: int) -> dict[str, float]:
            """Process evaluation metrics for a single batch.

            Args:
                batch_idx: Index of the batch to process.

            Returns:
                Dictionary containing metrics for this batch.
            """
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_data_batch))

            batch_metrics = []

            for i in range(start_idx, end_idx):
                pred_idx, probs = all_predictions[i]
                ground_truth = ground_truth_batch[i]

                if ground_truth not in species_to_idx:
                    metrics = {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "top_1": 0.0,
                        "top_5": 0.0,
                        "roc_auc": 0.0,
                    }
                    batch_metrics.append(metrics)
                    continue

                true_label = species_to_idx[ground_truth]

                correct = true_label == pred_idx
                accuracy = 1.0 if correct else 0.0
                precision = 1.0 if correct else 0.0
                recall = 1.0 if correct else 0.0
                f1 = 1.0 if correct else 0.0

                top_k = min(5, self.num_classes)
                top_probs, top_indices = torch.topk(probs, top_k)

                top_1 = 1.0 if true_label == top_indices[0].item() else 0.0
                top_5 = 1.0 if true_label in top_indices.cpu().numpy() else 0.0

                try:
                    y_true_bin = label_binarize(
                        [true_label],
                        classes=range(self.num_classes),
                    )
                    y_score = probs.cpu().numpy().reshape(1, -1)
                    roc_auc = roc_auc_score(y_true_bin, y_score, multi_class="ovr")
                except (ValueError, Exception):
                    roc_auc = 0.0

                metrics = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "top_1": top_1,
                    "top_5": top_5,
                    "roc_auc": roc_auc,
                }
                batch_metrics.append(metrics)

            return self._aggregate_metrics(batch_metrics)

        num_eval_batches = (len(input_data_batch) + batch_size - 1) // batch_size
        batch_indices = range(num_eval_batches)

        all_metrics = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_evaluation_batch, idx) for idx in batch_indices]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating classification batches",
            ):
                try:
                    batch_result = future.result()
                    all_metrics.append(batch_result)
                except Exception as e:
                    print(f"Error processing evaluation batch: {e}")

        final_metrics = self._aggregate_metrics(all_metrics)

        y_true, y_pred = [], []
        for i in range(len(input_data_batch)):
            gt = ground_truth_batch[i]
            if gt in species_to_idx:
                true_label = species_to_idx[gt]
                pred_label = all_predictions[i][0]
                y_true.append(true_label)
                y_pred.append(pred_label)

        if y_true and y_pred:
            conf_matrix = confusion_matrix(
                y_true,
                y_pred,
                labels=range(self.num_classes),
            )
            final_metrics["confusion_matrix"] = conf_matrix.tolist()

        return final_metrics

    def _create_cls_learner(self, model_path: str) -> Any:
        """Create a FastAI learner for classification.

        Creates a new FastAI vision learner with the specified architecture and
        configuration parameters. Used when pre-trained model loading fails.

        Args:
            model_path: Path where the trained model will be saved.

        Returns:
            FastAI vision learner instance.

        Raises:
            Exception: If learner creation fails due to missing data or config issues.
        """
        if self.data_dir is None or not self.data_dir.exists():
            raise Exception(
                "Cannot create learner: data directory not specified or doesn't exist",
            )

        model = timm.create_model(
            self.arch,
            pretrained=True,
            num_classes=self.num_classes,
        )

        params = self.config.classifier.params
        input_size = params.input_size

        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=parent_label,
            splitter=RandomSplitter(valid_pct=0.2),
            item_tfms=Resize(input_size),
            batch_tfms=aug_transforms(size=input_size),
        )

        batch_size = getattr(self.config.training, "batch_size", 32)
        dls = dblock.dataloaders(self.data_dir, bs=batch_size)

        metrics = getattr(self.config.training, "metrics", ["accuracy"])
        learn = vision_learner(
            dls,
            self.arch,
            metrics=metrics,
            loss_func=CrossEntropyLossFlat(),
            model=model,
        )

        learn.export(model_path)
        print(f"New learner created and saved to: {model_path}")

        return learn

    def get_species_names(self) -> list[str]:
        """Get list of all species names in order of class indices.

        Returns:
            List of species names ordered by class index.
        """
        return [self.species_map[i] for i in sorted(self.species_map.keys())]

    def get_class_index(self, species_name: str) -> int | None:
        """Get class index for a given species name.

        Args:
            species_name: Name of the species.

        Returns:
            Class index if species exists, None otherwise.
        """
        return self.settings.species_config.get_index_by_species(species_name)

