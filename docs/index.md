# CulicidaeLab Documentation

Welcome to the CulicidaeLab documentation! This library provides tools for mosquito (Culicidae) image analysis and machine learning.

## Core Modules

### [Configuration Management](config_manager.md)
Centralized configuration system using OmegaConf for managing application settings.

### [Resource Management](resource_manager.md)
Handles resource directories and file management across different operating systems.

### [Base Predictor](base_predictor.md)
Abstract base class for all prediction models in the library.

### [Loader Protocol](loader_protocol.md)
Defines the interface for data loading components.

### [Settings](settings.md)
Application-wide settings and constants.

### [Species Configuration](species_config.md)
Configuration and utilities for mosquito species handling.

### [Utilities](utils.md)
Common utility functions used throughout the library.

## Dataset Modules

### [Datasets Manager](datasets/manager.md)
Centralized management of mosquito datasets.

### [HuggingFace Integration](datasets/huggingface.md)
Integration with HuggingFace datasets for mosquito data.

## Prediction Modules

### [Classifier](predictors/classifier.md)
Image classification models for mosquito species identification.

### [Detector](predictors/detector.md)
Object detection models for mosquito localization.

### [Segmenter](predictors/segmenter.md)
Image segmentation models for detailed mosquito analysis.

### [Model Weights Manager](predictors/weights_manager.md)
Handles downloading and managing model weights.

## Provider Modules

### [HuggingFace Provider](providers/huggingface.md)
Integration with HuggingFace models and datasets.

### [Kaggle Provider](providers/kaggle.md)
Access to Kaggle datasets and competitions.

### [Roboflow Provider](providers/roboflow.md)
Integration with Roboflow for computer vision tasks.

### [Remote URL Provider](providers/remote_url.md)
Handles downloading resources from remote URLs.

## Getting Started

### Installation
```bash
pip install culicidaelab
```

### Basic Usage
```python
from culicidaelab import ConfigManager

# Initialize configuration
config = ConfigManager().load_config()

# Use other modules...
```

## Contributing
Contributions are welcome! Please see our [contributing guide](CONTRIBUTING.md) for more information.

## License
[Specify License]
