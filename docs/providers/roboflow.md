# Roboflow Provider

## Overview
The `RoboflowProvider` class provides seamless integration with Roboflow's computer vision platform, enabling easy downloading and management of datasets in various formats (YOLO, COCO, etc.).

## Features

- **Dataset Downloading**: Download datasets in multiple formats
- **Version Management**: Access specific dataset versions
- **Metadata Retrieval**: Get detailed dataset information
- **Authentication**: Secure API key management
- **Format Conversion**: Automatic conversion between dataset formats

## Installation

```bash
pip install culicidaelab roboflow
```

## Configuration

### Environment Variables

```bash
export ROBOFLOW_API_KEY="your_roboflow_api_key"
export ROBOFLOW_WORKSPACE="your_workspace"
export ROBOFLOW_PROJECT="your_project"
export ROBOFLOW_VERSION=1
```

### Config File

```yaml
providers:
  roboflow:
    api_key: "your_roboflow_api_key"
    workspace: "your_workspace"
    dataset: "your_dataset"
    version: 1
    format: "yolov8"  # or "coco", "pascal", etc.
```

## Usage

### Initialization

```python
from culicidaelab.core import ConfigManager
from culicidaelab.providers.roboflow_provider import RoboflowProvider

# Initialize configuration
config_manager = ConfigManager()

# Create provider instance
provider = RoboflowProvider(config_manager)
```

### Downloading Datasets

```python
# Download dataset with default settings
dataset_path = provider.download()

# Download specific split
dataset_path = provider.download(split="train")  # or "valid", "test"

# Download with custom format
dataset_path = provider.download(format="coco")
```

### Getting Dataset Metadata

```python
# Get metadata
metadata = provider.get_metadata()
print(f"Dataset name: {metadata['name']}")
print(f"Number of images: {metadata['splits']['train']}")

# List available versions
versions = provider.list_versions()
for version in versions['versions']:
    print(f"Version {version['version']}: {version['name']}")
```

## API Reference

### RoboflowProvider Class

#### Constructor

```python
RoboflowProvider(config_manager: ConfigManager)
```

**Parameters:**

- `config_manager`: Instance of ConfigManager with provider configuration

#### Methods

##### `download(dataset_id: str | None = None, save_dir: str | None = None, split: str = "train", **kwargs) -> Path`

Download a dataset from Roboflow.

**Parameters:**

- `dataset_id`: Dataset identifier (default: from config)
- `save_dir`: Directory to save the dataset (default: from config)
- `split`: Dataset split to download ('train', 'valid', or 'test')
- `**kwargs`: Additional download parameters

**Returns:**

- Path to the downloaded dataset

##### `get_metadata(dataset_id: str | None = None) -> dict[str, Any]`

Get metadata for a dataset.

**Parameters:**
- `dataset_id`: Dataset identifier (default: from config)

**Returns:**

- Dictionary containing dataset metadata

##### `list_versions(dataset_id: str | None = None) -> dict[str, Any]`

List all versions of a dataset.

**Parameters:**
- `dataset_id`: Dataset identifier (default: from config)

**Returns:**

- Dictionary containing version information

## Advanced Usage

### Custom Dataset Format

```python
# Download in COCO format
dataset_path = provider.download(format="coco")

# Download in Pascal VOC format
dataset_path = provider.download(format="pascal")
```

### Using with Custom Configuration

```python
# Update config
config = config_manager.get_config()
config.providers.roboflow.format = "yolov5"
config_manager.update_config(config)

# Now downloads will use YOLOv5 format
dataset_path = provider.download()
```

## Error Handling

### Handling Authentication Errors

```python
try:
    provider = RoboflowProvider(config_manager)
except ValueError as e:
    print(f"Authentication failed: {e}")
    print("Please set the ROBOFLOW_API_KEY environment variable.")
```

### Handling Download Failures

```python
try:
    dataset_path = provider.download(dataset_id="nonexistent/dataset")
except Exception as e:
    print(f"Failed to download dataset: {e}")
```

## Best Practices

### Version Control

- Pin dataset versions for reproducibility
- Document dataset versions in your project

### Dataset Organization

- Use consistent directory structures
- Keep raw and processed data separate

### Format Selection

- Choose the format that best fits your model
- Consider conversion overhead for large datasets

## Troubleshooting

### Common Issues

#### Authentication Failures

- Verify the API key is correct
- Check workspace and project names
- Ensure the key has necessary permissions

#### Dataset Not Found

- Verify dataset name and workspace
- Check if the dataset is private
- Ensure you have access to the dataset

#### Download Issues

- Check your internet connection
- Verify Roboflow service status
- Look for rate limiting

## Integration

The RoboflowProvider integrates with:

- Computer vision pipelines
- Model training scripts
- Data preprocessing workflows
- Experiment tracking systems
