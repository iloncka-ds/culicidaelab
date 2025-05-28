# Mosquito Classifier

## Overview
The `MosquitoClassifier` class provides functionality for classifying mosquito species from images using deep learning models. It leverages FastAI and timm (PyTorch Image Models) to support a wide range of model architectures for image classification tasks.

## Features
- **Multiple Model Architectures**: Supports any model available in the timm library
- **Species Classification**: Classifies mosquito species from input images
- **Batch Processing**: Efficient processing of multiple images
- **Visualization**: Built-in visualization of classification results
- **Evaluation**: Comprehensive model evaluation metrics
- **Cross-Platform**: Works on both Windows and POSIX systems

## Installation
```bash
pip install culicidaelab timm fastai
```

## Quick Start

### Basic Usage
```python
from culicidaelab.predictors.classifier import MosquitoClassifier
from culicidaelab.core import ConfigManager

# Initialize configuration manager
config_manager = ConfigManager()

# Initialize classifier
classifier = MosquitoClassifier(
    model_path="path/to/model.pkl",
    config_manager=config_manager
)

# Load an image (numpy array in RGB format)
import cv2
image = cv2.imread("mosquito.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get predictions
predictions = classifier.predict(image)
print(f"Top prediction: {predictions[0][0]} with confidence {predictions[0][1]:.2f}")
```

## API Reference

### `MosquitoClassifier` Class

#### Constructor
```python
MosquitoClassifier(
    model_path: str | Path,
    config_manager: ConfigManager,
    load_model: bool = False
)
```

**Parameters**:
- `model_path`: Path to pre-trained model weights (.pkl file)
- `config_manager`: Instance of `ConfigManager` containing model settings
- `load_model`: Whether to load the model immediately (default: False)

#### Methods

##### `predict(input_data: np.ndarray) -> list[tuple[str, float]]`
Classify mosquito species in an image.

**Parameters**:
- `input_data`: Input image as numpy array in RGB format (H, W, 3)

**Returns**:
- List of (species_name, confidence) tuples, sorted by confidence

##### `predict_batch(images: list[np.ndarray]) -> list[list[tuple[str, float]]]`
Classify multiple images in a batch.

**Parameters**:
- `images`: List of input images as numpy arrays

**Returns**:
- List of prediction lists, one per image

##### `evaluate(test_data: Any) -> dict[str, float]`
Evaluate the classifier on test data.

**Parameters**:
- `test_data`: Test dataset in FastAI format

**Returns**:
- Dictionary of evaluation metrics

##### `visualize(image: np.ndarray, prediction: tuple[str, float]) -> np.ndarray`
Visualize the classification result on the input image.

**Parameters**:
- `image`: Input image as numpy array
- `prediction`: Prediction tuple (species_name, confidence)

**Returns**:
- Image with visualization overlay

## Configuration

The classifier requires a configuration with the following structure:

```yaml
classifier:
  model_arch: "resnet50"  # timm model architecture
  img_size: 224  # Input image size
  batch_size: 32  # Batch size for inference
  
  # Training parameters (if training)
  lr: 1e-3
  epochs: 10
  
  # Data augmentation
  augment: True
  item_tfms: [Resize(460)]
  batch_tfms: [*aug_transforms(size=224)]
  
  # Species mapping
  species_classes: "path/to/species_classes.yaml"
```

## Advanced Usage

### Training a New Model
```python
from culicidaelab.predictors.classifier import MosquitoClassifier
from culicidaelab.core import ConfigManager

# Initialize with configuration
config_manager = ConfigManager()
classifier = MosquitoClassifier("new_model.pkl", config_manager)

# Train the model
metrics = classifier.train(
    train_data="path/to/train",
    valid_data="path/to/valid",
    epochs=10
)

# Save the trained model
classifier.save_model("trained_model.pkl")
```

### Using Custom Data Augmentation
```python
from fastai.vision.all import *

# Define custom transforms
custom_tfms = aug_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=20.0,
    max_zoom=1.1,
    max_lighting=0.2,
    max_warp=0.2,
)

# Update config with custom transforms
config = config_manager.get_config()
config.classifier.batch_tfms = custom_tfms
config_manager.update_config(config)

# Initialize classifier with custom transforms
classifier = MosquitoClassifier("model.pkl", config_manager)
```

## Best Practices

1. **Model Selection**:
   - Choose appropriate model architecture based on available compute resources
   - Larger models (e.g., ResNet152) offer better accuracy but require more memory
   - Smaller models (e.g., MobileNetV3) are faster but may have lower accuracy

2. **Data Preparation**:
   - Ensure consistent image sizes across the dataset
   - Normalize images according to model requirements
   - Use appropriate data augmentation to prevent overfitting

3. **Performance Optimization**:
   - Use batch processing for multiple images
   - Enable GPU acceleration when available
   - Consider model quantization for deployment on edge devices

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify the model file exists and is not corrupted
   - Check compatibility between model architecture and timm version
   - Ensure the configuration matches the model's expected input format

2. **Low Accuracy**
   - Check for class imbalance in the training data
   - Increase model capacity or add more training data
   - Adjust learning rate and other hyperparameters

3. **Memory Issues**
   - Reduce batch size
   - Use smaller image dimensions
   - Enable mixed precision training

## Integration with Other Components

The `MosquitoClassifier` integrates with:
- `ConfigManager` for configuration management
- `ResourceManager` for model and data loading
- Data preprocessing pipelines
- Visualization tools

---
*Documentation generated on: 2025-05-28*
