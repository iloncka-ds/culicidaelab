"""
Module for mosquito species classification using FastAI and timm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np
import timm
import torch
import torch.nn as nn
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.all import aug_transforms
from fastai.vision.all import CategoryBlock
from fastai.vision.all import create_body
from fastai.vision.all import DataBlock
from fastai.vision.all import get_image_files
from fastai.vision.all import ImageBlock
from fastai.vision.all import imagenet_stats
from fastai.vision.all import Normalize
from fastai.vision.all import parent_label
from fastai.vision.all import RandomSplitter
from fastai.vision.all import Resize
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from .settings import SpeciesConfig

# Standard Library

# Third Party
# Local


class MosquitoClassifier:
    """Class for classifying mosquito species using FastAI and timm models."""

    def __init__(
        self,
        model_path: str | None = None,
        arch: str = "convnext_base",
        config_path: str | None = None,
        data_dir: str | None = None,
    ) -> None:
        """
        Initialize the mosquito classifier.

        Args:
            model_path (str, optional): Path to pre-trained model weights
            arch (str): Architecture name from timm library
            config_path (str, optional): Path to species configuration file
            data_dir (str, optional): Path to data directory to infer species from structure
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch = arch
        self.data_dir = data_dir

        # Initialize species configuration
        self.species_config = SpeciesConfig(config_path, data_dir)
        self.species_map = self.species_config.get_species_map()
        self.num_classes = self.species_config.get_num_species()

        # Create model using fastai and timm
        self.learn = self._create_learner(model_path)

    def _create_learner(self, model_path: str | None = None) -> Learner:
        """
        Create a FastAI learner with timm backbone.

        Args:
            model_path: Path to saved model weights

        Returns:
            fastai.Learner: FastAI learner object
        """
        if not hasattr(self, "data_dir") or not self.data_dir:
            raise ValueError("data_dir must be provided to create a learner")

        path = Path(self.data_dir)
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")

        # Create dataloaders
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(),
            get_y=parent_label,
            item_tfms=Resize(224),
            batch_tfms=[
                *aug_transforms(size=224, min_scale=0.75),
                Normalize.from_stats(*imagenet_stats),
            ],
        ).dataloaders(path, bs=16)

        # Create model using timm
        arch = self.arch
        base_model = timm.create_model(arch, pretrained=True)
        model = create_body(base_model)
        model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Linear(model[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

        # Create learner
        learn = Learner(
            dls,
            model,
            loss_func=CrossEntropyLossFlat(),
            metrics=[accuracy_score],
        )

        # Load weights if provided
        if model_path and Path(model_path).exists():
            learn.load(model_path)

        learn.model.to(self.device)
        learn.model.eval()
        return learn

    def classify(
        self,
        image: str | np.ndarray,
    ) -> list[tuple[str, float]]:
        """
        Classify mosquito species in an image.

        Args:
            image: Input image (file path or numpy array)

        Returns:
            List of tuples (species_name, confidence_score)
        """
        # Convert numpy array to PIL Image if necessary
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)

        # Predict using fastai
        pred, pred_idx, probs = self.learn.predict(image)

        # Get top predictions
        top_k = min(5, self.num_classes)
        top_probs, top_indices = torch.topk(probs, top_k)

        predictions = []
        for idx, prob in zip(top_indices.cpu().numpy(), top_probs.cpu().numpy()):
            if idx in self.species_map:
                predictions.append((self.species_map[idx], float(prob)))
            else:
                predictions.append((f"Unknown Species {idx}", float(prob)))

        return predictions

    def train(
        self,
        data_path: str,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, float]:
        """
        Train the model using FastAI.

        Args:
            data_path: Path to training data directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training

        Returns:
            Dictionary containing training metrics
        """
        # Update species mapping from training data directory
        self.species_config.infer_from_directory(data_path)
        self.species_map = self.species_config.get_species_map()
        self.num_classes = self.species_config.get_num_species()

        # Create proper DataLoaders
        dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(),
            get_y=parent_label,
            item_tfms=Resize(224),
            batch_tfms=[
                *aug_transforms(size=224, min_scale=0.75),
                Normalize.from_stats(*imagenet_stats),
            ],
        ).dataloaders(Path(data_path), bs=batch_size)

        # Update learner with new data
        self.learn.dls = dls

        # Recreate the model with the correct number of classes
        self.learn.model = self._create_model()

        # Train the model
        self.learn.fine_tune(epochs, learning_rate)

        # Save the species configuration
        config_path = Path(data_path) / "species_config.yaml"
        self.species_config.save_to_file(config_path)

        # Get training metrics
        metrics = {
            "train_loss": float(self.learn.recorder.losses[-1]),
            "valid_loss": float(self.learn.recorder.values[-1][0]),
            "accuracy": float(self.learn.recorder.values[-1][1]),
        }

        return metrics

    def save_model(self, save_path: str) -> None:
        """
        Save the model weights and species configuration.

        Args:
            save_path (str): Path to save model weights
        """
        save_path_obj = Path(save_path)
        self.learn.save(save_path)

        # Save species configuration alongside the model
        config_path = save_path_obj.parent / f"{save_path_obj.stem}_species_config.yaml"
        self.species_config.save_to_file(config_path)

    def update_species_map(self, new_species_map: dict[int, str]) -> None:
        """
        Update the species mapping dictionary.

        Args:
            new_species_map: Dictionary mapping class indices to species names
        """
        self.species_map.update(new_species_map)
        self.species_config.species_map = self.species_map
        self.num_classes = len(self.species_map)

    def _create_model(self) -> nn.Module:
        """
        Create a new model with the correct number of classes.

        Returns:
            nn.Module: PyTorch model
        """
        arch = self.arch
        base_model = timm.create_model(arch, pretrained=True)
        model = create_body(base_model)
        model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Linear(model[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )
        return model

    def evaluate(
        self,
        y_true: np.ndarray | torch.Tensor | list,
        y_pred: np.ndarray | torch.Tensor | list,
        classes: int | list[Any] | dict[Any, int] | None = None,
        threshold: float = 0.5,
        average: str = "macro",
    ) -> Mapping[str, float | list[list[int]] | dict[str, dict[str, float]]]:
        """
        Evaluate classification model performance.

        Args:
            y_true: Ground truth labels
            y_pred: Model predictions
            classes: Optional number of classes or class mapping
            threshold: Classification threshold (default: 0.5)
            average: Averaging method for multi-class metrics (default: 'macro')

        Returns:
            Dictionary containing metrics (accuracy, precision, recall, f1, etc.)

        Raises:
            ValueError: If input arrays are empty or have different lengths
        """
        # Convert inputs to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)

        # Check for empty arrays
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Check for length mismatch
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length")

        # Determine class information
        if classes is None:
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            classes = list(range(len(unique_classes)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
        elif isinstance(classes, int):
            classes = list(range(classes))
            label_to_idx = {i: i for i in classes}
        elif isinstance(classes, dict):
            label_to_idx = classes
            classes = list(label_to_idx.values())
        else:
            label_to_idx = {label: idx for idx, label in enumerate(classes)}

        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        num_classes = len(classes)
        is_binary = num_classes == 2

        # Convert string labels to indices if needed
        if not isinstance(y_true[0], (int, np.integer)):
            y_true = np.array([label_to_idx[label] for label in y_true])
        if not isinstance(y_pred[0], (int, np.integer)) and y_pred.ndim == 1:
            y_pred = np.array([label_to_idx[label] for label in y_pred])

        # Handle different prediction formats
        if y_pred.ndim == 2:  # Probabilities for each class
            if is_binary:
                y_pred = y_pred[:, 1]  # Take probability of positive class
                y_pred_binary = (y_pred > threshold).astype(int)
                y_pred_labels = y_pred_binary
            else:
                y_pred_labels = np.argmax(y_pred, axis=1)
                y_pred_binary = label_binarize(y_pred_labels, classes=classes)
        else:
            y_pred_labels = y_pred
            if is_binary:
                y_pred_binary = (y_pred > threshold).astype(int)
            else:
                y_pred_binary = label_binarize(y_pred, classes=classes)

        # Calculate metrics
        accuracy = float(accuracy_score(y_true, y_pred_labels))
        metrics: dict[str, float | list[list[int]] | dict[str, dict[str, float]]] = {
            "accuracy": accuracy,
        }

        # Handle single-class case
        if len(np.unique(y_true)) == 1:
            per_class_metrics: dict[str, dict[str, float]] = {
                idx_to_label[classes[0]]: {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                },
            }
            metrics.update(
                {
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1": 1.0,
                    "per_class": per_class_metrics,
                    "average_precision": 1.0,
                    "average_recall": 1.0,
                    "average_f1": 1.0,
                    "confusion_matrix": confusion_matrix(y_true, y_pred_labels).tolist(),
                },
            )
            return metrics

        if is_binary:
            # Binary classification metrics
            cm = confusion_matrix(y_true, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics.update(
                {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "true_negatives": int(tn),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                },
            )

            # Only calculate ROC AUC if there are two classes
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_binary))
        else:
            # Multi-class metrics
            y_true_binary = label_binarize(y_true, classes=classes)

            # Calculate per-class metrics
            per_class_metrics = {}
            for i, class_label in enumerate(classes):
                class_true = y_true_binary[:, i]
                class_pred = y_pred_binary[:, i]
                cm = confusion_matrix(class_true, class_pred)
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                per_class_metrics[idx_to_label[i]] = {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }

            # Calculate average metrics
            metrics.update(
                {
                    "confusion_matrix": confusion_matrix(y_true, y_pred_labels).tolist(),
                    "per_class": per_class_metrics,
                    "average_precision": float(np.mean([m["precision"] for m in per_class_metrics.values()])),
                    "average_recall": float(np.mean([m["recall"] for m in per_class_metrics.values()])),
                    "average_f1": float(np.mean([m["f1"] for m in per_class_metrics.values()])),
                },
            )

            # Only calculate ROC AUC if there are multiple classes
            if len(np.unique(y_true)) > 1:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true_binary, y_pred_binary, average=average, multi_class="ovr"),
                )

        return metrics
