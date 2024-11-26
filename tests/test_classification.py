"""
Tests for the classification module.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

from culicidaelab.classification import MosquitoClassifier
from culicidaelab.settings import SpeciesConfig


@pytest.fixture
def sample_species_config():
    """Create a sample species configuration for testing."""
    species_map = {
        "Aedes aegypti": 0,
        "Anopheles gambiae": 1,
        "Culex quinquefasciatus": 2,
    }
    return SpeciesConfig(species_map=species_map)


@pytest.fixture
def sample_classifier(tmp_path):
    """Create a sample classifier instance for testing."""
    # Create a dummy model file
    model_path = tmp_path / "dummy_model.pth"
    torch.save({"state_dict": {}}, model_path)

    # Create a sample data directory with dummy images
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for species in ["Aedes aegypti", "Anopheles gambiae", "Culex quinquefasciatus"]:
        species_dir = data_dir / species
        species_dir.mkdir()
        # Create a dummy image
        img = Image.new("RGB", (224, 224), color="white")
        img.save(species_dir / "sample.jpg")

    return MosquitoClassifier(
        model_path=str(model_path),
        data_dir=str(data_dir),
    )


def test_classifier_initialization(sample_classifier):
    """Test classifier initialization."""
    assert isinstance(sample_classifier, MosquitoClassifier)
    assert sample_classifier.num_classes == 3


def test_classifier_data_loading(sample_classifier, tmp_path):
    """Test data loading functionality."""
    data_dir = Path(sample_classifier.data_dir)
    assert data_dir.exists()
    assert len(list(data_dir.glob("*/*.jpg"))) == 3


def test_classifier_evaluate_binary():
    """Test evaluation metrics for binary classification."""
    classifier = MosquitoClassifier()

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

    # Test case 3: Mixed predictions
    y_pred = np.array([0, 0, 0, 1])
    metrics = classifier.evaluate(y_true, y_pred)
    assert 0.0 < metrics["accuracy"] < 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0


def test_classifier_evaluate_multiclass(sample_species_config):
    """Test evaluation metrics for multi-class classification."""
    classifier = MosquitoClassifier(config_path=None)
    classifier.num_classes = 3

    # Test case 1: Perfect predictions
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    metrics = classifier.evaluate(y_true, y_pred, classes=3)
    assert metrics["accuracy"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["average_recall"] == 1.0
    assert metrics["average_f1"] == 1.0

    # Test case 2: All wrong predictions
    y_pred = np.array([1, 2, 0, 1, 2, 0])
    metrics = classifier.evaluate(y_true, y_pred, classes=3)
    assert metrics["accuracy"] == 0.0
    assert metrics["average_precision"] == 0.0
    assert metrics["average_recall"] == 0.0
    assert metrics["average_f1"] == 0.0

    # Test case 3: Mixed predictions
    y_pred = np.array([0, 1, 1, 0, 1, 2])
    metrics = classifier.evaluate(y_true, y_pred, classes=3)
    assert 0.0 < metrics["accuracy"] < 1.0
    assert 0.0 <= metrics["average_precision"] <= 1.0
    assert 0.0 <= metrics["average_recall"] <= 1.0
    assert 0.0 <= metrics["average_f1"] <= 1.0


def test_classifier_evaluate_edge_cases():
    """Test evaluation metrics with edge cases."""
    classifier = MosquitoClassifier()

    # Test case 1: Empty inputs
    with pytest.raises(ValueError):
        classifier.evaluate([], [])

    # Test case 2: Mismatched lengths
    with pytest.raises(ValueError):
        classifier.evaluate([0, 1], [0])

    # Test case 3: Invalid class labels
    with pytest.raises(ValueError):
        classifier.evaluate([0, 1], [0, 2], classes=2)

    # Test case 4: Invalid probabilities
    with pytest.raises(ValueError):
        classifier.evaluate([0, 1], [1.5, -0.5])


def test_classifier_with_torch_tensors():
    """Test evaluation with PyTorch tensors."""
    classifier = MosquitoClassifier()

    # Binary classification
    y_true = torch.tensor([0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 0, 1])
    metrics = classifier.evaluate(y_true, y_pred)
    assert metrics["accuracy"] == 1.0

    # Multi-class classification
    y_true = torch.tensor([0, 1, 2])
    y_pred = torch.tensor([0, 1, 2])
    metrics = classifier.evaluate(y_true, y_pred, classes=3)
    assert metrics["accuracy"] == 1.0


def test_classifier_with_class_mapping():
    """Test evaluation with class name mapping."""
    classifier = MosquitoClassifier()
    class_map = {"cat": 0, "dog": 1, "bird": 2}

    y_true = ["cat", "dog", "bird", "cat"]
    y_pred = ["cat", "dog", "bird", "cat"]
    metrics = classifier.evaluate(y_true, y_pred, classes=class_map)
    assert metrics["accuracy"] == 1.0

    # Test with wrong predictions
    y_pred = ["dog", "bird", "cat", "dog"]
    metrics = classifier.evaluate(y_true, y_pred, classes=class_map)
    assert metrics["accuracy"] == 0.0


def test_classifier_prediction_probabilities():
    """Test evaluation with prediction probabilities."""
    classifier = MosquitoClassifier()

    # Binary classification with probabilities
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    metrics = classifier.evaluate(y_true, y_pred)
    assert metrics["accuracy"] == 1.0

    # Multi-class with probabilities
    y_true = np.array([0, 1, 2])
    y_pred = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.15, 0.8],
        ],
    )
    metrics = classifier.evaluate(y_true, y_pred, classes=3)
    assert metrics["accuracy"] == 1.0
