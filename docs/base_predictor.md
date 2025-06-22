# Base Predictor Module

## Overview
The `base_predictor` module provides an abstract base class for all prediction models in the culicidaelab library. It defines a common interface that all predictors (classifiers, detectors, segmenters) must implement, ensuring consistency across different model types.

## Features
- **Unified Interface**: Standardized methods for model prediction and evaluation
- **Batch Processing**: Built-in support for processing batches of inputs
- **Parallel Execution**: Utilizes thread pools for efficient batch processing
- **Evaluation Framework**: Standardized approach to model evaluation
- **Visualization Support**: Common interface for prediction visualization

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from culicidaelab.core import BasePredictor, ConfigManager
import numpy as np

# Initialize configuration
config_manager = ConfigManager()

# Create a predictor (abstract - must be subclassed)
class MyPredictor(BasePredictor):
    def _load_model(self):
        # Implement model loading
        pass

    def predict(self, input_data):
        # Implement prediction logic
        pass

    def visualize(self, input_data, predictions, save_path=None):
        # Implement visualization
        pass

    def evaluate(self, input_data, ground_truth):
        # Implement evaluation
        pass

# Initialize predictor
predictor = MyPredictor(
    model_path="path/to/model.pt",
    config_manager=config_manager,
    load_model=True
)

# Make predictions
input_data = np.random.rand(224, 224, 3)  # Example input
predictions = predictor.predict(input_data)
```

## API Reference

### BasePredictor

#### Constructor
```python
BasePredictor(
    model_path: str | Path,
    config_manager: ConfigManager,
    load_model: bool = False
)
```

**Parameters**:
- `model_path`: Path to the model weights
- `config_manager`: Configuration manager instance
- `load_model`: If True, loads model immediately. If False, delays loading until needed.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `model_path` | Path | Path to the model weights |
| `model_loaded` | bool | Whether the model is currently loaded |

#### Methods

##### `predict(input_data: np.ndarray) -> Any`
Make predictions on the input data.

**Parameters**:
- `input_data`: Input data as a numpy array

**Returns**:
- Predictions in a format specific to the predictor type

##### `evaluate(input_data: np.ndarray, ground_truth: Any) -> dict[str, float]`
Evaluate model predictions against ground truth.

**Parameters**:
- `input_data`: Input data to evaluate on
- `ground_truth`: Ground truth annotations

**Returns**:
- Dictionary of evaluation metrics

##### `evaluate_batch(input_data_batch, ground_truth_batch, num_workers=4, batch_size=32) -> dict[str, float]`
Evaluate model on a batch of inputs in parallel.

**Parameters**:
- `input_data_batch`: List of input data arrays
- `ground_truth_batch`: List of ground truth annotations
- `num_workers`: Number of parallel workers
- `batch_size`: Size of batches to process

**Returns**:
- Aggregated evaluation metrics

##### `visualize(input_data: np.ndarray, predictions: Any, save_path: str | Path | None = None) -> np.ndarray`
Visualize predictions on the input data.

**Parameters**:
- `input_data`: Original input data
- `predictions`: Model predictions
- `save_path`: Optional path to save visualization

**Returns**:
- Visualization as a numpy array

##### `predict_batch(input_data_batch, num_workers=4, batch_size=32) -> list[Any]`
Make predictions on a batch of inputs in parallel.

**Parameters**:
- `input_data_batch`: List of input data arrays
- `num_workers`: Number of parallel workers
- `batch_size`: Size of batches to process

**Returns**:
- List of predictions

## Abstract Methods

Child classes must implement these methods:

### `_load_model() -> None`
Load the model from the specified path.

### `_aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]`
Aggregate metrics from multiple evaluations.

**Parameters**:
- `metrics_list`: List of metric dictionaries

**Returns**:
- Aggregated metrics dictionary

## Advanced Usage

### Custom Evaluation Metrics
```python
class CustomPredictor(BasePredictor):
    # ... other methods ...

    def evaluate(self, input_data, ground_truth):
        predictions = self.predict(input_data)

        # Calculate custom metrics
        accuracy = self._calculate_accuracy(predictions, ground_truth)
        precision = self._calculate_precision(predictions, ground_truth)
        recall = self._calculate_recall(predictions, ground_truth)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': 2 * (precision * recall) / (precision + recall + 1e-7)
        }
```

### Custom Batch Processing
```python
class CustomPredictor(BasePredictor):
    # ... other methods ...

    def predict_batch(self, input_data_batch, num_workers=4, batch_size=32):
        # Custom batch processing logic
        results = []
        for i in range(0, len(input_data_batch), batch_size):
            batch = input_data_batch[i:i+batch_size]
            # Process batch with custom logic
            batch_results = self._custom_batch_predict(batch)
            results.extend(batch_results)
        return results
```

## Best Practices

1. **Model Loading**:
   - Implement lazy loading in `_load_model()` to improve initialization time
   - Handle model download if weights aren't available locally

2. **Error Handling**:
   - Validate input shapes and types in predict()
   - Provide meaningful error messages for invalid inputs

3. **Performance**:
   - Use batch processing for better GPU utilization
   - Implement `_predict_batch()` for optimized batch inference

4. **Memory Management**:
   - Clear CUDA cache if using GPU
   - Use generators for large datasets

## Integration with Other Modules

The BasePredictor integrates with:
- `ConfigManager` for configuration management
- `ResourceManager` for model and data paths
- Dataset loaders for input processing
- Visualization utilities for result interpretation

## Example Implementations

### Image Classifier
```python
class ImageClassifier(BasePredictor):
    def _load_model(self):
        self.model = load_pretrained_model(self.model_path)
        self.model.eval()
        self.classes = self.model.get_classes()
        self.model_loaded = True

    def predict(self, image):
        if not self.model_loaded:
            self._load_model()

        # Preprocess image
        tensor = preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        return {
            'class': self.classes[torch.argmax(probs)],
            'confidence': torch.max(probs).item(),
            'probabilities': probs.numpy()
        }
```

## Performance Considerations

1. **Batch Size**:
   - Larger batches improve GPU utilization but increase memory usage
   - Tune based on available GPU memory

2. **Number of Workers**:
   - For CPU-bound tasks, match number of workers to CPU cores
   - For I/O-bound tasks, increase workers beyond core count

3. **Memory Usage**:
   - Clear CUDA cache between batches if needed
   - Use mixed precision for faster inference with minimal accuracy loss

## Contributing

When implementing a new predictor:
1. Inherit from BasePredictor
2. Implement all abstract methods
3. Add type hints and docstrings
4. Include unit tests
5. Document any additional dependencies

---
*Documentation generated on: 2025-05-28*
