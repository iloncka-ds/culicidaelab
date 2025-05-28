# Model Weights Manager

## Overview
The `ModelWeightsManager` class provides a centralized way to manage model weights in the culicidaelab library. It handles downloading, caching, and providing access to model weights, with support for both local and remote (Hugging Face Hub) weight files.

## Features
- **Automatic Download**: Downloads weights on-demand if not found locally
- **Caching**: Caches downloaded weights for future use
- **Multiple Model Types**: Supports different types of models (detection, segmentation, classification)
- **Hugging Face Integration**: Seamlessly downloads weights from Hugging Face Hub
- **Configurable Paths**: Customizable local storage locations

## Installation
```bash
pip install culicidaelab huggingface-hub
```

## Quick Start

### Basic Usage
```python
from culicidaelab.core import ConfigManager
from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

# Initialize configuration manager
config_manager = ConfigManager()

# Initialize weights manager
weights_manager = ModelWeightsManager(config_manager)

# Get path to model weights
# This will prompt for download if weights aren't found locally
try:
    weights_path = weights_manager.get_weights("detection")
    print(f"Weights path: {weights_path}")
except Exception as e:
    print(f"Error: {e}")
```

## API Reference

### `ModelWeightsManager` Class

#### Constructor
```python
ModelWeightsManager(config_manager: ConfigManager)
```

**Parameters**:
- `config_manager`: Instance of `ConfigManager` containing model weights configuration

#### Methods

##### `get_weights(model_type: str) -> str`
Get the path to model weights, downloading them if necessary.

**Parameters**:
- `model_type`: Type of model ('detection', 'segmentation', or 'classification')

**Returns**:
- Absolute path to the model weights file

**Raises**:
- `ValueError`: If the model type is not recognized
- `FileNotFoundError`: If weights aren't found and download is declined
- `Exception`: For download or file operation errors

## Configuration

The weights manager requires a configuration with the following structure:

```yaml
models:
  weights:
    detection:
      local_path: "models/weights/detection.pt"  # Relative to root_dir
      remote_repo: "organization/detection-model"
      remote_file: "detection.pt"
    classification:
      local_path: "models/weights/classification.pt"
      remote_repo: "organization/classification-model"
      remote_file: "classification.pt"
    segmentation:
      local_path: "models/weights/segmentation.pt"
      remote_repo: "organization/segmentation-model"
      remote_file: "segmentation.pt"

paths:
  root_dir: "."  # Base directory for relative paths
```

## Advanced Usage

### Customizing Download Location
```python
from pathlib import Path

# Get the default config
config = config_manager.get_config()

# Update the local path for detection model
config.models.weights.detection.local_path = "custom/path/detection.pt"

# Update the configuration
config_manager.update_config(config)

# Now the weights will be downloaded to the custom location
weights_path = weights_manager.get_weights("detection")
```

### Silent Mode (Non-Interactive)
```python
import os

# Set environment variable to run in non-interactive mode
os.environ["CULICIDAELAB_NON_INTERACTIVE"] = "1"

try:
    # This will raise an exception if weights aren't found
    weights_path = weights_manager.get_weights("detection")
except FileNotFoundError as e:
    print(f"Weights not found and download was not attempted: {e}")
```

## Best Practices

1. **Version Control**:
   - Add large model weights to `.gitignore`
   - Document the expected checksums for model weights
   - Use versioned model repositories on Hugging Face Hub

2. **Caching**:
   - Store weights in a centralized location
   - Use symbolic links if you need to reference weights from multiple locations
   - Consider using a shared network location for team environments

3. **Error Handling**:
   - Always handle the case when weights aren't available
   - Provide clear error messages to users about how to obtain the weights
   - Include fallback behavior when possible

## Troubleshooting

### Common Issues

1. **Download Permission Denied**
   - Check write permissions for the target directory
   - Ensure sufficient disk space is available
   - Verify network connectivity to Hugging Face Hub

2. **Model Version Mismatch**
   - Ensure the model architecture matches the weights
   - Check for version compatibility between the code and weights
   - Update to the latest version of the model repository if available

3. **Slow Downloads**
   - Use a stable internet connection
   - Consider downloading weights manually and placing them in the expected location
   - Use a download manager for large files

## Integration with Other Components

The `ModelWeightsManager` integrates with:
- `ConfigManager` for configuration management
- Hugging Face Hub for remote weights
- Various model classes for weight loading
- Training and inference pipelines

---
*Documentation generated on: 2025-05-29*
