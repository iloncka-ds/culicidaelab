"""Tests for the top-level serve function."""

import pytest
from unittest.mock import patch, MagicMock

from culicidaelab.serve import serve, clear_serve_cache


@pytest.fixture(autouse=True)
def clear_cache_after_test():
    """Fixture to automatically clear the cache after each test."""
    yield
    clear_serve_cache()


@patch("culicidaelab.serve.get_settings")
@patch("culicidaelab.serve.MosquitoClassifier")
def test_serve_classifier_initialization(mock_classifier, mock_get_settings):
    """Test that serve initializes the classifier correctly on first call."""
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    mock_predictor_instance = MagicMock()
    mock_classifier.return_value = mock_predictor_instance

    serve("fake_image.jpg", predictor_name="classifier")

    mock_get_settings.assert_called_once()
    mock_classifier.assert_called_once_with(mock_settings, mode="serve")
    mock_predictor_instance.predict.assert_called_once_with("fake_image.jpg")


@patch("culicidaelab.serve.get_settings")
@patch("culicidaelab.serve.MosquitoDetector")
def test_serve_detector_with_kwargs(mock_detector, mock_get_settings):
    """Test that serve passes kwargs to the detector's predict method."""
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    mock_predictor_instance = MagicMock()
    mock_detector.return_value = mock_predictor_instance

    serve("image.png", predictor_name="detector", confidence_threshold=0.8)

    mock_detector.assert_called_once_with(mock_settings, mode="serve")
    mock_predictor_instance.predict.assert_called_once_with(
        "image.png",
        confidence_threshold=0.8,
    )


@patch("culicidaelab.serve.get_settings")
@patch("culicidaelab.serve.MosquitoClassifier")
def test_serve_caching_behavior(mock_classifier, mock_get_settings):
    """Test that predictor instances are cached after the first call."""
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    mock_predictor_instance = MagicMock()
    mock_classifier.return_value = mock_predictor_instance

    # First call - should initialize and cache
    serve("image1.jpg", predictor_name="classifier")
    assert mock_classifier.call_count == 1
    assert mock_predictor_instance.predict.call_count == 1

    # Second call - should use cached instance
    serve("image2.jpg", predictor_name="classifier")
    assert mock_classifier.call_count == 1  # No new initialization
    assert mock_predictor_instance.predict.call_count == 2


def test_serve_unknown_predictor():
    """Test that serve raises a ValueError for an unknown predictor name."""
    with pytest.raises(ValueError, match="Unknown predictor_name: 'invalid_predictor'"):
        serve("image.jpg", predictor_name="invalid_predictor")


@patch("culicidaelab.serve.get_settings")
@patch("culicidaelab.serve.MosquitoSegmenter")
def test_clear_serve_cache(mock_segmenter, mock_get_settings):
    """Test that clear_serve_cache removes cached predictors."""
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    mock_predictor_instance = MagicMock()
    mock_segmenter.return_value = mock_predictor_instance

    # Initialize and cache
    serve("image.jpg", predictor_name="segmenter")
    assert mock_segmenter.call_count == 1

    # Clear the cache
    clear_serve_cache()

    # This call should re-initialize
    serve("image.jpg", predictor_name="segmenter")
    assert mock_segmenter.call_count == 2
    mock_predictor_instance.unload_model.assert_called_once()
