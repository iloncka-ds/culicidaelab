# HuggingFace Provider

## Overview
The `HuggingFaceProvider` class enables interaction with the Hugging Face Hub for downloading datasets and model weights. It's designed to work seamlessly with the Hugging Face ecosystem.

## Features
- **Dataset Downloading**: Download and manage datasets from Hugging Face Hub
- **Model Weights**: Handle model weight downloads and management
- **Authentication**: Secure API key management
- **Metadata Retrieval**: Get detailed information about datasets

## Installation
```bash
pip install culicidaelab datasets huggingface_hub
```

## Configuration

### Environment Variables
```bash
export HUGGINGFACE_API_KEY="your_api_key_here"
```

### Config File
```yaml
providers:
  huggingface:
    api_key: "your_api_key_here"
    provider_url: "https://huggingface.co/api/datasets/{dataset_name}"
```

## Usage

### Initialization
```python
from culicidaelab.core import ConfigManager
from culicidaelab.providers.huggingface_provider import HuggingFaceProvider

# Initialize configuration
config_manager = ConfigManager()

# Create provider instance
provider = HuggingFaceProvider(config_manager)
```

### Downloading Datasets
```python
# Download a dataset
dataset_path = provider.download_dataset("username/dataset_name")
print(f"Dataset downloaded to: {dataset_path}")

# Download with specific configuration or split
dataset_path = provider.download_dataset(
    "username/dataset_name",
    split="train",
    config_name="default"
)
```

### Getting Dataset Metadata
```python
# Get dataset metadata
metadata = provider.get_dataset_metadata("username/dataset_name")
print(f"Dataset description: {metadata.get('description')}")
print(f"Number of examples: {metadata.get('downloads')}")
```

### Downloading Model Weights
```python
# Download model weights for a specific task
weights_path = provider.download_model_weights("classification")
print(f"Weights downloaded to: {weights_path}")
```

## Advanced Usage

### Using with Datasets Library
```python
from datasets import load_dataset

# Load dataset directly using the downloaded path
dataset = load_dataset(str(dataset_path))
```

### Custom Save Directory
```python
# Specify custom save directory
custom_path = "path/to/save/dataset"
dataset_path = provider.download_dataset("username/dataset_name", save_dir=custom_path)
```

## Error Handling

### Handling Missing API Key
```python
try:
    provider = HuggingFaceProvider(config_manager)
except ValueError as e:
    print(f"Error: {e}")
    print("Please set the HUGGINGFACE_API_KEY environment variable.")
```

### Handling Download Failures
```python
try:
    dataset_path = provider.download_dataset("nonexistent/dataset")
except Exception as e:
    print(f"Failed to download dataset: {e}")
```

## Best Practices

1. **API Key Management**
   - Store API keys in environment variables or a secure secrets manager
   - Never commit API keys to version control

2. **Caching**
   - The HuggingFace datasets library automatically caches downloaded files
   - Clear cache when needed using `datasets.builder.DatasetBuilder.clear_cache()`

3. **Streaming**
   - For large datasets, use the streaming mode to avoid downloading the entire dataset
   ```python
   from datasets import load_dataset
   dataset = load_dataset("username/large_dataset", streaming=True)
   ```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify the API key is correctly set
   - Check for any typos in the key
   - Ensure the key has the necessary permissions

2. **Dataset Not Found**
   - Verify the dataset name and organization
   - Check if the dataset is private (requires authentication)

3. **Connection Issues**
   - Check your internet connection
   - Verify you can access https://huggingface.co
   - Check for any firewall restrictions

## Integration

The HuggingFaceProvider integrates with:
- `ConfigManager` for configuration
- HuggingFace datasets library
- Model training pipelines
- Caching systems
