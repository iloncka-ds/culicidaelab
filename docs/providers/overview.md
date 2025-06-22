# Providers Module

## Overview
The providers module in culicidaelab offers a flexible way to interact with various data sources and services. It follows a provider pattern, allowing for easy extension and integration with different platforms.

## Available Providers

1. **HuggingFace Provider**
   - Manages datasets and model weights from Hugging Face Hub
   - Supports dataset downloading and model weight management

2. **Kaggle Provider**
   - Facilitates downloading datasets from Kaggle
   - Requires Kaggle API credentials

3. **Remote URL Provider**
   - Handles downloading files from any public URL
   - Supports resumable downloads and progress tracking

4. **Roboflow Provider**
   - Manages computer vision datasets from Roboflow
   - Supports various dataset formats (YOLO, COCO, etc.)

## Common Interface

All providers implement the `BaseProvider` interface, which defines the following core methods:

- `download_dataset()`: Downloads a dataset from the provider
- `get_metadata()`: Retrieves metadata about a dataset
- `get_provider_name()`: Returns the name of the provider

## Configuration

Providers are configured through the main application configuration. Each provider may require specific API keys or settings:

```yaml
providers:
  huggingface:
    api_key: "your_hf_api_key"
    provider_url: "https://huggingface.co/api/datasets/{dataset_name}"

  kaggle:
    api_key: "your_kaggle_api_key"
    username: "your_kaggle_username"

  roboflow:
    api_key: "your_roboflow_api_key"
    workspace: "your_workspace"
    dataset: "dataset_name"
    project_version: 1
```

## Usage Example

```python
from culicidaelab.core import ConfigManager
from culicidaelab.providers import HuggingFaceProvider

# Initialize configuration
config_manager = ConfigManager()

# Create a provider
provider = HuggingFaceProvider(config_manager)

# Download a dataset
dataset_path = provider.download_dataset("username/dataset_name")
```

## Best Practices

1. **API Keys**:
   - Store API keys in environment variables or a secure secrets manager
   - Never commit API keys to version control

2. **Caching**:
   - Implement caching to avoid unnecessary downloads
   - Check for existing files before downloading

3. **Error Handling**:
   - Handle API rate limits and timeouts gracefully
   - Provide meaningful error messages for authentication failures

## Extending with Custom Providers

To create a new provider:

1. Create a new class that inherits from `BaseProvider`
2. Implement the required methods
3. Register the provider in the provider factory
4. Add configuration options to the main config

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Verify API keys are correctly set in the environment
   - Check for any IP restrictions on the API key

2. **Download Failures**
   - Verify network connectivity
   - Check if the resource exists and is accessible
   - Look for rate limiting or quota issues

3. **Memory Issues**
   - For large datasets, use streaming or chunked downloads
   - Monitor memory usage during large downloads

## Integration with Other Components

The providers module integrates with:
- `ConfigManager` for configuration
- Dataset loaders for data processing
- Model training pipelines
- Caching systems for performance optimization
