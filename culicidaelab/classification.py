"""
Module for mosquito species classification using FastAI and timm.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from typing import Any
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
from sklearn.metrics import accuracy_score, confusion_matrix, label_binarize, roc_auc_score

from .config import SpeciesConfig

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
        # Create dummy data to initialize the learner
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
        ).dataloaders(Path("."), bs=16)

        # Create model using timm
        arch = self.arch
        model = create_body(partial(timm.create_model, arch, pretrained=True))
        model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Linear(model[-1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_classes),
        )

        # Create learner
        learn = Learner(dls, model, loss_func=CrossEntropyLossFlat())

        if model_path:
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
        model = create_body(partial(timm.create_model, arch, pretrained=True))
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
    ) -> dict[str, Any]:
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

        # Determine class information
        if classes is None:
            classes = [0, 1]  # Binary classification
        elif isinstance(classes, int):
            classes = list(range(classes))
        elif isinstance(classes, dict):
            idx_to_label = {v: k for k, v in classes.items()}
            classes = list(classes.values())
        else:
            label_to_idx = {label: idx for idx, label in enumerate(classes)}
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        num_classes = len(classes)
        is_binary = num_classes == 2

        # Convert string labels to indices if needed
        if not isinstance(y_true[0], (int, np.integer)):
            y_true = np.array([label_to_idx[label] for label in y_true])

        # Handle different prediction formats
        if y_pred.ndim == 2:  # Probabilities for each class
            if is_binary:
                y_pred = y_pred[:, 1]  # Take probability of positive class
            else:
                y_pred_binary = label_binarize(np.argmax(y_pred, axis=1), classes=classes)
                y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            if is_binary:
                y_pred_binary = (y_pred > threshold).astype(int)
                y_pred_labels = y_pred_binary
            else:
                y_pred_binary = label_binarize(y_pred, classes=classes)
                y_pred_labels = y_pred

        # Calculate metrics
        accuracy = float(accuracy_score(y_true, y_pred_labels))

        if is_binary:
            # Binary classification metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

            return {
                "accuracy": accuracy,
                "tpr": float(tpr),
                "tnr": float(tnr),
                "precision": float(precision),
                "recall": float(tpr),
                "f1": float(f1),
                "roc_auc": float(roc_auc_score(y_true, y_pred)),
            }
        else:
            # Multi-class metrics
            per_class_metrics = {}
            y_true_bin = label_binarize(y_true, classes=classes)

            for idx, class_label in enumerate(classes):
                class_true = y_true_bin[:, idx]
                class_pred = y_pred_binary[:, idx]
                tn, fp, fn, tp = confusion_matrix(class_true, class_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

                label = idx_to_label[idx] if "idx_to_label" in locals() else idx
                per_class_metrics[label] = {
                    "tpr": float(tpr),
                    "tnr": float(tnr),
                    "precision": float(precision),
                    "recall": float(tpr),
                    "f1": float(f1),
                }

            return {
                "accuracy": accuracy,
                "per_class": per_class_metrics,
                "average_precision": float(np.mean([m["precision"] for m in per_class_metrics.values()])),
                "average_recall": float(np.mean([m["recall"] for m in per_class_metrics.values()])),
                "average_f1": float(np.mean([m["f1"] for m in per_class_metrics.values()])),
                "roc_auc": float(roc_auc_score(y_true_bin, y_pred, multi_class="ovr", average=average)),
            }
