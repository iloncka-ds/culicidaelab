# HuggingFace Dataset Loader

## Overview
The `HuggingFaceDatasetLoader` is an implementation of the `DatasetLoader` protocol that provides seamless integration with the Hugging Face 🤗 Datasets library. It allows you to easily load and work with datasets from the Hugging Face Hub or local files in your culicidaelab projects.

## Features
- **Hugging Face Hub Integration**: Load datasets directly from the Hugging Face Hub
- **Unified Interface**: Implements the standard `DatasetLoader` protocol
- **Flexible Loading**: Supports various dataset formats and configurations
- **Streaming Support**: Efficiently work with large datasets using streaming

## Installation
```bash
pip install culicidaelab datasets
```

## Quick Start

### Basic Usage
```python
from culicidaelab.datasets.huggingface import HuggingFaceDatasetLoader

# Initialize the loader
loader = HuggingFaceDatasetLoader()

# Load a dataset from the Hugging Face Hub
dataset = loader.load_dataset("entomopraxis/mosquito-species", split="train")

# Load a local dataset
dataset = loader.load_dataset("path/to/local/dataset", split="train")
```

## API Reference

### `HuggingFaceDatasetLoader` Class

#### Constructor
```python
HuggingFaceDatasetLoader()
```

#### Methods

##### `load_dataset(path: str, split: str | None = None, **kwargs) -> Any`
Load a dataset using the Hugging Face datasets library.

**Parameters**:
- `path` (str): Path to the dataset (can be a Hugging Face Hub dataset ID or local path)
- `split` (str, optional): Which split of the data to load (e.g., 'train', 'test', 'validation')
- `**kwargs`: Additional arguments to pass to `datasets.load_dataset()`

**Returns**:
- The loaded dataset object

**Raises**:
- `datasets.DatasetNotFoundError`: If the dataset cannot be found
- `Exception`: For other loading errors

## Advanced Usage

### Using with Custom Configurations
```python
loader = HuggingFaceDatasetLoader()

# Load a specific configuration of a dataset
dataset = loader.load_dataset(
    "entomopraxis/mosquito-species",
    split="train",
    name="high_quality"  # Specific configuration name
)
```

### Streaming Large Datasets
```python
loader = HuggingFaceDatasetLoader()

# Stream a large dataset without downloading it entirely
dataset = loader.load_dataset(
    "big_dataset_repo",
    split="train",
    streaming=True  # Enable streaming mode
)

# Process the dataset in a memory-efficient way
for example in dataset:
    # Process each example
    process(example)
```

### Using with Custom Loading Options
```python
loader = HuggingFaceDatasetLoader()

# Load dataset with custom options
dataset = loader.load_dataset(
    "entomopraxis/mosquito-species",
    split="train",
    use_auth_token=True,  # For private datasets
    num_proc=4,  # Use 4 processes for loading
    cache_dir="./custom_cache"  # Custom cache directory
)
```

## Example: Complete Workflow

```python
from culicidaelab.datasets.huggingface import HuggingFaceDatasetLoader
from culicidaelab.datasets import DatasetsManager
from culicidaelab.core import ConfigManager

# Initialize components
config_manager = ConfigManager()
loader = HuggingFaceDatasetLoader()

# Create datasets manager
datasets_manager = DatasetsManager(
    config_manager=config_manager,
    dataset_loader=loader,
    config_path="configs/datasets.yaml"
)

# Add a dataset configuration
datasets_manager.add_dataset(
    "mosquito_train",
    {
        "path": "entomopraxis/mosquito-species",
        "split": "train",
        "description": "Training split of mosquito species dataset"
    }
)

# Load the dataset
dataset = datasets_manager.load_dataset("mosquito_train")

# Use the dataset
for example in dataset:
    image = example["image"]
    label = example["label"]
    # Process the example
```

## Best Practices

1. **Caching**:
   - Hugging Face datasets are cached by default in `~/.cache/huggingface/datasets`
   - Use the `cache_dir` parameter to specify a custom cache location
   - Clear the cache periodically to save disk space

2. **Streaming for Large Datasets**:
   - Enable `streaming=True` for large datasets that don't fit in memory
   - Process data in batches when using streaming

3. **Authentication**:
   - Use `use_auth_token=True` for private datasets
   - Store authentication tokens securely using environment variables

4. **Error Handling**:
   - Handle dataset loading errors gracefully
   - Check for dataset availability before loading

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Verify the dataset ID or path is correct
   - Check if you have the necessary permissions for private datasets
   - Ensure you're connected to the internet

2. **Memory Issues**
   - Use `streaming=True` for large datasets
   - Increase system swap space if needed
   - Process data in smaller batches

3. **Version Conflicts**
   - Ensure compatibility between the `datasets` library and your Python environment
   - Check for updates to the dataset repository

## Integration with Other Components

The `HuggingFaceDatasetLoader` can be used with:
- `DatasetsManager` for centralized dataset management
- Model training pipelines
- Data preprocessing and augmentation
- Evaluation and metrics calculation

---
*Documentation generated on: 2025-05-28*
