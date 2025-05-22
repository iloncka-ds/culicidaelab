"""
Module for mosquito species classification using FastAI and timm.
"""

from __future__ import annotations

import os
from typing import Any
import numpy as np
import timm
import torch
from fastai.vision.all import (
    aug_transforms,
    CategoryBlock,
    DataBlock,
    get_image_files,
    ImageBlock,
    parent_label,
    RandomSplitter,
    Resize,
    vision_learner,
    CrossEntropyLossFlat,
    load_learner,
)
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from culicidaelab.species_classes_manager import SpeciesClassesManager
from culicidaelab.core._base_predictor import BasePredictor
from culicidaelab.core.config_manager import ConfigManager
from contextlib import contextmanager
import pathlib
import platform
import cv2


@contextmanager
def set_posix_windows():
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
    """Class for classifying mosquito species using FastAI and timm models."""

    def __init__(self, model_path: str | os.path, config_manager: ConfigManager) -> None:
        """
        Initialize the mosquito classifier.

        Args:
            model_path: Path to pre-trained model weights
            config_manager: Configuration manager instance
        """
        super().__init__(model_path, config_manager)
        self.config = self.config_manager.get_config()

        # Set up model architecture and paths
        self.arch = self.config.model.arch
        self.data_dir = self.config.data.data_dir

        # Initialize species configuration
        self.species_config = SpeciesClassesManager(config_manager=self.config_manager)
        self.species_map = self.species_config.get_species_map()
        self.num_classes = len(self.species_map)
        print(f"Number of species classes: {self.num_classes}")

    def _load_model(self) -> None:
        """Load the FastAI model with timm backbone."""
        with set_posix_windows():
            try:
                self.learner = load_learner(self.model_path)
            except Exception:
                # If loading fails, create new learner
                self.learner = self._create_cls_learner(self.model_path)

    def predict(self, input_data: np.ndarray) -> list[tuple[str, float]]:
        """
        Classify mosquito species in an image.

        Args:
            input_data: Input image as numpy array

        Returns:
            List of tuples (species_name, confidence_score)
        """
        if not self.model_loaded:
            self.load_model()

        # Convert numpy array to PIL Image
        if input_data.dtype == np.uint8:
            image = Image.fromarray(input_data)
        else:
            image = Image.fromarray((input_data * 255).astype(np.uint8))

        # Get predictions
        pred_class, pred_idx, probabilities = self.learner.predict(image)

        # Convert predictions to list of (species, confidence) tuples
        species_probs = []
        for idx, prob in enumerate(probabilities):
            species_name = self.species_map.get(idx, f"unknown_{idx}")
            species_probs.append((species_name, float(prob)))

        # Sort by confidence
        species_probs.sort(key=lambda x: x[1], reverse=True)

        # Return top predictions based on config
        top_k = self.config.model.top_k
        return species_probs[:top_k]

    def visualize(
        self,
        input_data: np.ndarray,
        predictions: list[tuple[str, float]],
        save_path: str | os.path | None = None,
    ) -> np.ndarray:
        """
        Visualize classification predictions on the image.

        Args:
            input_data: Original image
            predictions: List of (species_name, confidence) tuples
            save_path: Optional path to save visualization

        Returns:
            np.ndarray: Image with visualized predictions
        """
        # Create visualization
        vis_img = input_data.copy()

        # Add text for each prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.visualization.font_scale
        thickness = self.config.visualization.text_thickness
        color = self.config.visualization.text_color

        y_offset = 30
        for species, conf in predictions:
            text = f"{species}: {conf:.2f}"
            cv2.putText(vis_img, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 30

        if save_path:
            cv2.imwrite(str(save_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

        return vis_img

    def evaluate(
        self,
        input_data: np.ndarray,
        ground_truth: str,
    ) -> dict[str, float]:
        """
        Evaluate classification predictions against ground truth.

        Args:
            input_data: Input image
            ground_truth: Ground truth species name

        Returns:
            Dictionary containing accuracy and confidence metrics
        """
        predictions = self.predict(input_data)
        pred_species = predictions[0][0] if predictions else "unknown"
        confidence = predictions[0][1] if predictions else 0.0

        correct = pred_species == ground_truth

        return {
            "accuracy": float(correct),
            "confidence": float(confidence),
            "top_1_correct": float(correct),
            "top_5_correct": float(any(p[0] == ground_truth for p in predictions[:5])),
        }

    def _create_cls_learner(self, model_path: str) -> Any:
        """Create a FastAI learner for classification."""
        # Create model architecture
        model = timm.create_model(
            self.arch,
            pretrained=True,
            num_classes=self.num_classes,
        )

        # Create data block
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            get_y=parent_label,
            splitter=RandomSplitter(),
            item_tfms=Resize(self.config.model.input_size),
            batch_tfms=aug_transforms(
                size=self.config.model.input_size,
                min_scale=self.config.augmentation.min_scale,
                do_flip=self.config.augmentation.do_flip,
                flip_vert=self.config.augmentation.flip_vert,
                max_rotate=self.config.augmentation.max_rotate,
                min_zoom=self.config.augmentation.min_zoom,
                max_zoom=self.config.augmentation.max_zoom,
                max_lighting=self.config.augmentation.max_lighting,
                max_warp=self.config.augmentation.max_warp,
            ),
        )

        # Create data loaders
        dls = dblock.dataloaders(
            self.data_dir,
            bs=self.config.training.batch_size,
        )

        # Create learner
        learn = vision_learner(
            dls,
            self.arch,
            metrics=self.config.training.metrics,
            loss_func=CrossEntropyLossFlat(),
            model=model,
        )

        # Save and return
        learn.export(model_path)
        return learn

    def evaluate_batch(
        self,
        input_data_batch: list[np.ndarray],
        ground_truth_batch: list[str],
        num_workers: int = 4,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """
        Optimized batch evaluation for classifier that processes predictions in batches.

        Args:
            input_data_batch: List of input images
            ground_truth_batch: List of ground truth species names
            num_workers: Number of parallel workers
            batch_size: Size of batches to process at once

        Returns:
            Dictionary containing aggregated evaluation metrics
        """
        if len(input_data_batch) != len(ground_truth_batch):
            raise ValueError(
                "Number of inputs must match number of ground truth annotations",
            )

        if len(input_data_batch) == 0:
            raise ValueError("Input batch cannot be empty")

        # Convert numpy arrays to PIL Images
        images = [Image.fromarray(img) for img in input_data_batch]

        # Get predictions for entire batch at once using FastAI's batch processing
        all_predictions = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                # Process batch predictions
                batch_preds = []
                for img in batch:
                    pred, pred_idx, probs = self.learner.predict(img)
                    batch_preds.append((pred_idx, probs))
                all_predictions.extend(batch_preds)

        # Process evaluation in parallel
        def process_batch(batch_idx: int) -> dict[str, float]:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_data_batch))

            batch_metrics = []
            for i in range(start_idx, end_idx):
                pred_idx, probs = all_predictions[i]
                ground_truth = ground_truth_batch[i]

                # Convert ground truth to numerical label
                if ground_truth in self.species_map.values():
                    true_label = [k for k, v in self.species_map.items() if v == ground_truth][0]
                else:
                    # If ground truth species is not in our map, count as incorrect
                    metrics = {
                        "accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "top_1": 0.0,
                        "top_5": 0.0,
                    }
                    batch_metrics.append(metrics)
                    continue

                # Get top predictions
                top_k = min(5, self.num_classes)
                top_probs, top_indices = torch.topk(probs, top_k)

                # Calculate metrics
                pred_label = pred_idx.item()
                accuracy = 1.0 if true_label == pred_label else 0.0
                precision = 1.0 if true_label == pred_label else 0.0
                recall = 1.0 if true_label == pred_label else 0.0
                f1 = 1.0 if true_label == pred_label else 0.0

                # Calculate top-k accuracy
                top_1 = 1.0 if true_label == top_indices[0].item() else 0.0
                top_5 = 1.0 if true_label in top_indices.cpu().numpy() else 0.0

                # For multi-class ROC-AUC, we need to binarize the labels
                y_true_bin = label_binarize(
                    [true_label],
                    classes=range(self.num_classes),
                )
                try:
                    roc_auc = roc_auc_score(
                        y_true_bin,
                        probs.cpu().numpy().reshape(1, -1),
                        multi_class="ovr",
                    )
                except ValueError:
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

        # Calculate number of batches
        num_batches = (len(input_data_batch) + batch_size - 1) // batch_size
        batch_indices = range(num_batches)

        # Process batches in parallel
        all_metrics = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_batch, idx) for idx in batch_indices]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating classification batches",
            ):
                try:
                    batch_result = future.result()
                    all_metrics.append(batch_result)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

        # Aggregate metrics from all batches
        final_metrics = self._aggregate_metrics(all_metrics)

        # Add confusion matrix for the entire batch
        y_true = []
        y_pred = []
        for i in range(len(input_data_batch)):
            gt = ground_truth_batch[i]
            if gt in self.species_map.values():
                true_label = [k for k, v in self.species_map.items() if v == gt][0]
                pred_label = all_predictions[i][0].item()
                y_true.append(true_label)
                y_pred.append(pred_label)

        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        final_metrics["confusion_matrix"] = conf_matrix.tolist()

        return final_metrics

    def _aggregate_metrics(
        self,
        metrics_list: list[dict[str, float]],
    ) -> dict[str, float]:
        """
        Aggregate metrics from multiple batches.

        Args:
            metrics_list: List of dictionaries containing metrics

        Returns:
            Dictionary containing aggregated metrics
        """
        if not metrics_list:
            return {}

        # Initialize aggregated metrics
        aggregated = {}

        # Get all metric keys except confusion_matrix
        metric_keys = [key for key in metrics_list[0].keys() if key != "confusion_matrix"]

        # Calculate mean and std for each metric
        for key in metric_keys:
            values = [m[key] for m in metrics_list if key in m]
            if values:
                aggregated[key] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        return aggregated
