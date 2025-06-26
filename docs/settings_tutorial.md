# Settings Module Tutorial

The settings module provides a centralized configuration management system for CulicidaeLab. This tutorial will show you how to use its main features.

## Basic Usage

```python
from culicidaelab.settings import get_settings

# Get settings instance with default configuration
settings = get_settings()

# Or specify a custom config directory
settings = get_settings(config_dir="path/to/config")
```

## Key Features

### 1. Model Weights Management

```python
# Get path to model weights for different model types
detection_weights = settings.get_model_weights_path("detection")
segmentation_weights = settings.get_model_weights_path("segmentation")
classification_weights = settings.get_model_weights_path("classification")
```

### 2. Dataset Management

```python
# Get dataset paths for different types
detection_dataset = settings.get_dataset_path("detection")
segmentation_dataset = settings.get_dataset_path("segmentation")
classification_dataset = settings.get_dataset_path("classification")
```

### 3. Directory Access

```python
# Access important directories
weights_dir = settings.weights_dir()  # Get weights directory
datasets_dir = settings.datasets_dir()  # Get datasets directory
cache_dir = settings.cache_dir()  # Get cache directory
config_dir = settings.config_dir()  # Get config directory
```

### 4. Processing Parameters

```python
# Get processing parameters for model inference
params = settings.get_processing_params()
```

## Environment Configuration

The settings module supports different environments (development, production, etc.):

1. Set environment through environment variable:
```bash
export APP_ENV=development  # or production
```

2. Or it defaults to "development" if not set

## Directory Structure

The settings module automatically manages several directories:
- `weights/`: For model weights storage
- `datasets/`: For dataset storage
- `cache/`: For temporary files
- `conf/`: For configuration files

## Configuration Files

Required configuration files in your config directory:
- `app_settings_{environment}.yaml`
- `models_config.yaml`
- `species_classes.yaml`
- `species_metadata.yaml`
- `datasets_config.yaml`

If any of these files are missing when using a custom config directory, they will be automatically copied from the default configuration.

## Best Practices

1. Always use the `get_settings()` function to obtain a settings instance
2. The Settings class is a singleton - multiple calls to `get_settings()` return the same instance
3. Use environment-specific configuration files for different deployment scenarios
4. Access paths through the settings API rather than constructing them manually
