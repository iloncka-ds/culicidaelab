# Datasets Manager

## Overview
The `DatasetsManager` class provides a unified interface for managing and loading datasets in the culicidaelab library. It works with any dataset loader that implements the `DatasetLoader` protocol, making it flexible for different data sources.

## Features
- **Unified Interface**: Standardized way to manage multiple datasets
- **Configuration Management**: Load and save dataset configurations
- **Lazy Loading**: Datasets are loaded only when needed
- **Flexible Integration**: Works with any dataset loader implementing the `DatasetLoader` protocol

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from culicidaelab.datasets import DatasetsManager
from culicidaelab.datasets.huggingface import HuggingFaceDatasetLoader
from culicidaelab.core import ConfigManager

# Initialize dependencies
config_manager = ConfigManager()
loader = HuggingFaceDatasetLoader()

# Create datasets manager
datasets_manager = DatasetsManager(
    config_manager=config_manager,
    dataset_loader=loader,
    datasets_dir="path/to/datasets",
    config_path="path/to/datasets_config.yaml"
)

# Load a dataset
dataset = datasets_manager.load_dataset("mosquito_detection", split="train")

# List available datasets
available_datasets = datasets_manager.list_datasets()
```

## API Reference

### `DatasetsManager` Class

#### Constructor
```python
DatasetsManager(
    config_manager: ConfigManager,
    dataset_loader: DatasetLoader,
    datasets_dir: str | Path | None = None,
    config_path: str | Path | None = None
)
```

**Parameters**:
- `config_manager`: Instance of `ConfigManager` for configuration handling
- `dataset_loader`: Dataset loader implementing the `DatasetLoader` protocol
- `datasets_dir`: Base directory for datasets (optional)
- `config_path`: Path to datasets configuration file (optional)

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `datasets_config` | `dict[str, Any]` | Dictionary containing dataset configurations |
| `loaded_datasets` | `dict[str, Any]` | Cache of loaded datasets |

#### Methods

##### `load_config() -> dict[str, Any]`
Load datasets configuration from the configured path.

**Returns**:
- Dictionary containing dataset configurations

**Raises**:
- `FileNotFoundError`: If the configuration file doesn't exist

##### `save_config(config_path: Path | None = None) -> None`
Save the current datasets configuration to a file.

**Parameters**:
- `config_path`: Optional path to save the configuration (uses instance path if None)

**Raises**:
- `ValueError`: If no configuration path is available

##### `get_dataset_info(dataset_name: str) -> dict[str, Any]`
Get information about a specific dataset.

**Parameters**:
- `dataset_name`: Name of the dataset

**Returns**:
- Dictionary containing dataset information

**Raises**:
- `KeyError`: If the dataset is not found

##### `list_datasets() -> list[str]`
List all available datasets in the configuration.

**Returns**:
- List of dataset names

##### `add_dataset(name: str, info: dict[str, Any]) -> None`
Add a new dataset to the configuration.

**Parameters**:
- `name`: Name of the dataset
- `info`: Dictionary containing dataset information

**Raises**:
- `ValueError`: If a dataset with the same name already exists

##### `remove_dataset(name: str) -> None`
Remove a dataset from the configuration.

**Parameters**:
- `name`: Name of the dataset to remove

**Raises**:
- `KeyError`: If the dataset is not found

##### `update_dataset(name: str, info: dict[str, Any]) -> None`
Update information for an existing dataset.

**Parameters**:
- `name`: Name of the dataset to update
- `info`: Dictionary containing updated dataset information

**Raises**:
- `KeyError`: If the dataset is not found

##### `load_dataset(dataset_name: str, split: str | None = None, **kwargs) -> Any`
Load a dataset using the configured dataset loader.

**Parameters**:
- `dataset_name`: Name of the dataset to load
- `split`: Optional dataset split to load (e.g., 'train', 'test')
- `**kwargs`: Additional arguments to pass to the dataset loader

**Returns**:
- The loaded dataset

**Raises**:
- `ValueError`: If the dataset path is not specified or loading fails

##### `get_loaded_dataset(dataset_name: str) -> Any`
Get a previously loaded dataset from the cache.

**Parameters**:
- `dataset_name`: Name of the dataset

**Returns**:
- The loaded dataset

**Raises**:
- `ValueError`: If the dataset has not been loaded yet

## Configuration File Format
The datasets configuration file should be in YAML format with the following structure:

```yaml
datasets:
  dataset1:
    path: path/to/dataset1
    type: custom_type
    description: Description of dataset1

  dataset2:
    path: path/to/dataset2
    type: another_type
    description: Description of dataset2

  # Additional datasets...
```

## Example: Creating a Custom Dataset Configuration

```python
from pathlib import Path
from culicidaelab.datasets import DatasetsManager
from culicidaelab.datasets.huggingface import HuggingFaceDatasetLoader
from culicidaelab.core import ConfigManager

# Initialize dependencies
config_manager = ConfigManager()
loader = HuggingFaceDatasetLoader()

# Create a new datasets manager
datasets_manager = DatasetsManager(
    config_manager=config_manager,
    dataset_loader=loader,
    datasets_dir="data/datasets"
)

# Add a new dataset
datasets_manager.add_dataset(
    "mosquito_species",
    {
        "path": "entomopraxis/mosquito-species",
        "type": "image_classification",
        "description": "Dataset containing mosquito species images",
        "classes": ["aedes_aegypti", "anopheles_gambiae", "culex_quinquefasciatus"]
    }
)

# Save the configuration
datasets_manager.save_config("configs/datasets.yaml")

# Later, load and use the dataset
datasets_manager = DatasetsManager(
    config_manager=config_manager,
    dataset_loader=loader,
    config_path="configs/datasets.yaml"
)

dataset = datasets_manager.load_dataset("mosquito_species", split="train")
```

## Best Practices

1. **Configuration Management**:
   - Store dataset configurations in version control
   - Use relative paths for portability
   - Document each dataset's structure and requirements

2. **Memory Management**:
   - Load only the necessary dataset splits
   - Use streaming for large datasets when possible
   - Clear unused datasets from memory

3. **Error Handling**:
   - Check for dataset existence before loading
   - Handle missing or corrupted data gracefully
   - Provide meaningful error messages

## Integration with Other Modules

The `DatasetsManager` integrates with:
- `ConfigManager` for configuration handling
- Any `DatasetLoader` implementation (e.g., `HuggingFaceDatasetLoader`)
- Model training and evaluation pipelines
- Data preprocessing components

## Performance Considerations

1. **Caching**:
   - The manager caches loaded datasets by default
   - Use `get_loaded_dataset()` to access cached datasets
   - Clear the cache when memory usage is a concern

2. **Lazy Loading**:
   - Datasets are loaded only when requested
   - Use the `split` parameter to load specific portions of a dataset

3. **Concurrent Loading**:
   - The manager is not thread-safe by default
   - Implement synchronization if accessing from multiple threads

## Troubleshooting

### Common Issues

1. **Configuration Errors**
   - Verify YAML syntax in configuration files
   - Check for missing required fields
   - Ensure paths are correctly specified

2. **Loading Failures**
   - Check network connectivity for remote datasets
   - Verify file permissions for local datasets
   - Ensure the dataset format matches the loader's expectations

3. **Memory Issues**
   - Use streaming for large datasets
   - Clear the dataset cache when not needed
   - Process data in smaller batches

---
*Documentation generated on: 2025-05-28*
