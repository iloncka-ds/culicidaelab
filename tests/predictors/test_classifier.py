import pytest
import numpy as np
from unittest.mock import Mock, patch
import torch
import sys
import types

sys.modules["culicidaelab.species_classes_manager"] = types.SimpleNamespace(
    SpeciesClassesManager=Mock(),
)

from culicidaelab.predictors.classifier import MosquitoClassifier


@pytest.fixture
def mock_config(tmp_path):
    config = Mock()
    config.classifier.model_arch = "resnet18"
    config.classifier.params.species_classes = tmp_path / "species.yaml"
    config.classifier.params.input_size = 224
    config.data.data_dir = str(tmp_path)
    config.training.batch_size = 2
    config.training.metrics = ["accuracy"]
    config.model.top_k = 2
    config.visualization.font_scale = 1.0
    config.visualization.text_thickness = 2
    config.visualization.text_color = (0, 255, 0)
    config.paths.root_dir = tmp_path
    return config


@pytest.fixture
def species_yaml(tmp_path):
    species_path = tmp_path / "species.yaml"
    species_path.write_text("0: species1\n1: species2\n2: species3\n")
    return species_path


@pytest.fixture
def mock_config_manager(mock_config):
    config_manager = Mock()
    config_manager.get_config.return_value = mock_config
    return config_manager


@pytest.fixture
def classifier(tmp_path, mock_config, mock_config_manager, species_yaml):
    with (
        patch(
            "culicidaelab.predictors.classifier.timm.create_model",
            return_value=Mock(),
        ),
        patch("culicidaelab.predictors.classifier.load_learner", return_value=Mock()),
        patch("culicidaelab.predictors.classifier.vision_learner", return_value=Mock()),
        patch("culicidaelab.predictors.classifier.get_image_files", return_value=[]),
    ):
        return MosquitoClassifier(
            model_path=tmp_path / "dummy.pkl",
            config_manager=mock_config_manager,
            load_model=False,
        )


def test_species_loading(classifier):
    assert classifier.num_classes == 3
    assert classifier.species_map[0] == "species1"
    assert classifier.species_map[1] == "species2"
    assert classifier.species_map[2] == "species3"


def test_predict(classifier):
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8)
    mock_probs = torch.tensor([0.7, 0.2, 0.1])
    classifier.learner = Mock()
    classifier.learner.predict.return_value = ("species1", 0, mock_probs)
    classifier.model_loaded = True
    results = classifier.predict(dummy_image)
    assert isinstance(results, list)
    assert results[0][0] == "species1"
    assert results[0][1] == pytest.approx(0.7, rel=1e-6)


def test_visualize(classifier):
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8)
    predictions = [("species1", 0.7), ("species2", 0.2)]
    vis_img = classifier.visualize(dummy_image, predictions)
    assert isinstance(vis_img, np.ndarray)
    assert vis_img.shape == dummy_image.shape


def test_evaluate(classifier):
    dummy_image = np.ones((224, 224, 3), dtype=np.uint8)
    with patch.object(classifier, "predict") as mock_predict:
        mock_predict.return_value = [
            ("species1", 0.7),
            ("species2", 0.2),
            ("species3", 0.1),
        ]
        metrics = classifier.evaluate(dummy_image, "species1")
        assert metrics["accuracy"] == 1.0
        assert metrics["top_1_correct"] == 1.0
        assert metrics["top_5_correct"] == 1.0
        assert metrics["confidence"] == 0.7


def test_evaluate_batch(classifier):
    dummy_images = [np.ones((224, 224, 3), dtype=np.uint8) for _ in range(2)]
    ground_truths = ["species1", "species1"]
    classifier.learner = Mock()
    classifier.learner.predict.return_value = (
        "species1",
        torch.tensor(0),
        torch.tensor([0.7, 0.2, 0.1]),
    )
    classifier.model_loaded = True
    results = classifier.evaluate_batch(
        dummy_images,
        ground_truths,
        batch_size=1,
        num_workers=1,
    )
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "confusion_matrix" in results


def test_get_species_names_and_index(classifier):
    names = classifier.get_species_names()
    assert names == ["species1", "species2", "species3"]
    assert classifier.get_class_index("species2") == 1
    assert classifier.get_class_index("not_a_species") is None
