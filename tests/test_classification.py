"""
Tests for the classification module.
"""

import pytest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastai.vision.learner import Learner

from culicidaelab.classifier import MosquitoClassifier
from culicidaelab.species_cofig import SpeciesConfig


class MockModel(nn.Module):
    """Mock model for testing."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


@pytest.fixture
def mock_learner(dummy_data_dir):
    """Create a mock learner for testing."""
    model = MockModel(num_classes=3)
    learner = MagicMock(spec=Learner)
    learner.model = model
    learner.dls = MagicMock()
    return learner


@pytest.fixture
def sample_species_config(tmp_path):
    """Create a sample species configuration for testing."""
    # Create a config file
    config_path = tmp_path / "species_config.yaml"
    with open(config_path, "w") as f:
        f.write(
            """
species_map:
  Aedes_aegypti: 0
  Anopheles_gambiae: 1
  Culex_quinquefasciatus: 2
""",
        )
    return SpeciesConfig(config_path=str(config_path))


@pytest.fixture
def dummy_data_dir(tmp_path):
    """Create a dummy data directory with minimal structure for testing."""
    data_dir = tmp_path / "dummy_data"
    data_dir.mkdir(exist_ok=True)

    # Create sample species directories with dummy images
    for species in ["Aedes_aegypti", "Anopheles_gambiae", "Culex_quinquefasciatus"]:
        species_dir = data_dir / species
        species_dir.mkdir(exist_ok=True)
        # Create a dummy image
        img = Image.new("RGB", (224, 224), color="white")
        img.save(species_dir / "dummy.jpg")

    return data_dir


@pytest.fixture
def sample_classifier(dummy_data_dir, mock_learner):
    """Create a sample classifier instance for testing."""
    # Create a dummy model file
    model_path = dummy_data_dir.parent / "dummy_model.pth"
    torch.save({"state_dict": {}}, model_path)

    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(
            model_path=str(model_path),
            data_dir=str(dummy_data_dir),
        )
        return classifier


def test_classifier_initialization(dummy_data_dir, mock_learner):
    """Test classifier initialization."""
    # Create a dummy model file
    model_path = dummy_data_dir.parent / "dummy_model.pth"
    torch.save({"state_dict": {}}, model_path)

    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(
            model_path=str(model_path),
            data_dir=str(dummy_data_dir),
        )
        assert isinstance(classifier, MosquitoClassifier)
        assert classifier.num_classes == 3
        assert len(classifier.species_map) == 3


def test_classifier_data_loading(sample_classifier, tmp_path):
    """Test data loading functionality."""
    data_dir = Path(sample_classifier.data_dir)
    assert data_dir.exists()
    assert len(list(data_dir.glob("*/*.jpg"))) == 3


def test_classifier_evaluate_binary(dummy_data_dir, mock_learner):
    """Test evaluation metrics for binary classification."""
    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(data_dir=str(dummy_data_dir))

        # Test case 1: Perfect predictions
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = classifier.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

        # Test case 2: All wrong predictions
        y_pred = np.array([1, 0, 1, 0])
        metrics = classifier.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0


def test_classifier_evaluate_multiclass(sample_species_config, dummy_data_dir, mock_learner):
    """Test evaluation metrics for multi-class classification."""
    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(data_dir=str(dummy_data_dir))
        species_map = {"Aedes_aegypti": 0, "Anopheles_gambiae": 1, "Culex_quinquefasciatus": 2}

        # Test case 1: Perfect predictions
        y_true = ["Aedes_aegypti", "Anopheles_gambiae", "Culex_quinquefasciatus"]
        y_pred = ["Aedes_aegypti", "Anopheles_gambiae", "Culex_quinquefasciatus"]

        # Convert species names to class indices
        y_true_idx = [species_map[name] for name in y_true]
        y_pred_idx = [species_map[name] for name in y_pred]

        metrics = classifier.evaluate(y_true_idx, y_pred_idx)
        assert metrics["accuracy"] == 1.0
        assert metrics["average_precision"] == 1.0
        assert metrics["average_recall"] == 1.0
        assert metrics["average_f1"] == 1.0

        # Test case 2: Mixed predictions
        y_true = ["Aedes_aegypti", "Anopheles_gambiae", "Culex_quinquefasciatus"]
        y_pred = ["Anopheles_gambiae", "Culex_quinquefasciatus", "Aedes_aegypti"]

        # Convert species names to class indices
        y_true_idx = [species_map[name] for name in y_true]
        y_pred_idx = [species_map[name] for name in y_pred]

        metrics = classifier.evaluate(y_true_idx, y_pred_idx)
        assert metrics["accuracy"] == 0.0
        assert metrics["average_precision"] < 1.0
        assert metrics["average_recall"] < 1.0
        assert metrics["average_f1"] < 1.0


def test_classifier_evaluate_edge_cases(dummy_data_dir, mock_learner):
    """Test evaluation metrics with edge cases."""
    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(data_dir=str(dummy_data_dir))

        # Test case 1: Single class
        y_true = np.array([0])
        y_pred = np.array([0])
        metrics = classifier.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 1.0

        # Test case 2: Empty arrays
        with pytest.raises(ValueError):
            classifier.evaluate(np.array([]), np.array([]))


def test_classifier_with_torch_tensors(dummy_data_dir, mock_learner):
    """Test evaluation with PyTorch tensors."""
    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(data_dir=str(dummy_data_dir))

        y_true = torch.tensor([0, 1, 2])
        y_pred = torch.tensor([0, 1, 2])
        metrics = classifier.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 1.0


def test_classifier_with_class_mapping(dummy_data_dir, mock_learner):
    """Test evaluation with class name mapping."""
    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(data_dir=str(dummy_data_dir))

        # Test with string labels
        y_true = ["Aedes_aegypti", "Anopheles_gambiae", "Culex_quinquefasciatus"]
        y_pred = ["Aedes_aegypti", "Anopheles_gambiae", "Culex_quinquefasciatus"]
        metrics = classifier.evaluate(y_true, y_pred)
        assert metrics["accuracy"] == 1.0


def test_classifier_prediction_probabilities(dummy_data_dir, mock_learner):
    """Test evaluation with prediction probabilities."""
    with patch("culicidaelab.classification.MosquitoClassifier._create_learner", return_value=mock_learner):
        classifier = MosquitoClassifier(data_dir=str(dummy_data_dir))

        # Test with probabilities
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]])
        metrics = classifier.evaluate(y_true, np.argmax(y_pred, axis=1))
        assert metrics["accuracy"] == 1.0
