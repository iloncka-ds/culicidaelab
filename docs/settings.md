# Settings Module

## Overview
The `settings` module provides a centralized way to manage application-wide settings and configurations in the culicidaelab library. It implements the singleton pattern to ensure consistent access to configuration values throughout the application.

## Features
- **Singleton Pattern**: Ensures a single source of truth for application settings
- **Hierarchical Configuration**: Supports multiple configuration sources with override capability
- **Environment-Aware**: Loads different configurations based on the environment (development, production, etc.)
- **Resource Management**: Provides access to application resource directories
- **Type Safety**: Uses Pydantic models for configuration validation

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from culicidaelab.core.settings import get_settings

# Get settings instance
settings = get_settings()

# Access configuration values
environment = settings.environment
root_dir = settings.root_dir

# Get resource directories
cache_dir = settings.get_resource_dir('cache_dir')
data_dir = settings.get_resource_dir('user_data_dir')
```

## API Reference

### `Settings` Class

#### Constructor
```python
Settings(config_dir: str | Path | None = None)
```

**Parameters**:
- `config_dir`: Optional path to an external configuration directory

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `environment` | str | Current application environment (e.g., 'development', 'production') |
| `root_dir` | Path | Root directory of the culicidaelab package |
| `config` | DictConfig | Access to the full configuration object |

#### Methods

##### `get_resource_dir(resource_type: str) -> Path`
Get a specific resource directory.

**Parameters**:
- `resource_type`: Type of resource directory (e.g., 'user_data_dir', 'cache_dir')

**Returns**:
- `Path`: Path to the requested resource directory

##### `get_dataset_path(dataset_type: str) -> Path`
Get the path to a specific dataset directory.

**Parameters**:
- `dataset_type`: Type of dataset ('detection', 'segmentation', or 'classification')

**Returns**:
- `Path`: Path to the dataset directory

##### `get_processing_params() -> dict[str, Any]`
Get processing parameters for model inference.

**Returns**:
- `dict`: Dictionary of processing parameters

### Module-Level Function

#### `get_settings(config_dir: str | Path | None = None) -> Settings`
Get or create a settings instance.

**Parameters**:
- `config_dir`: Optional path to an external configuration directory

**Returns**:
- `Settings`: The settings instance

## Configuration Structure

The settings module looks for configuration files in the following locations:

1. **Library Defaults**: `culicidaelab/conf/`
   - Contains default configuration files
   - Serves as a fallback for missing configurations

2. **User Configuration**: `~/.culicidaelab/` (default)
   - User-specific overrides
   - Created automatically if it doesn't exist

3. **Environment-Specific**: `app_settings/{environment}.yaml`
   - Environment-specific settings
   - Environment is determined by `APP_ENV` environment variable (default: 'development')

## Example Configuration

### Default Config Directory Structure
```
conf/
├── app_settings/
│   ├── development.yaml
│   └── production.yaml
├── datasets/
│   └── default.yaml
├── species/
│   ├── species_classes.yaml
│   └── species_metadata.yaml
└── config.yaml
```

### Example `config.yaml`
```yaml
datasets:
  paths:
    detection: datasets/detection
    segmentation: datasets/segmentation
    classification: datasets/classification

processing:
  image_size: [640, 640]
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]

resources:
  cache_dir: ${oc.env:CACHE_DIR, '~/.cache/culicidaelab'}
  model_dir: models
  dataset_dir: datasets
```

## Advanced Usage

### Using Custom Configuration Directory
```python
from pathlib import Path
from culicidaelab.core.settings import get_settings

# Load settings from a custom directory
custom_config_dir = Path("/path/to/custom/config")
settings = get_settings(config_dir=custom_config_dir)
```

### Accessing Nested Configuration
```python
# Get processing parameters
params = settings.get_processing_params()
image_size = params['image_size']
normalize = params['normalize']

# Access nested configuration
model_config = settings.config.model
learning_rate = model_config.training.learning_rate
batch_size = model_config.training.batch_size
```

## Best Practices

1. **Configuration Hierarchy**:
   - Library defaults (lowest priority)
   - User configuration (medium priority)
   - Environment variables (high priority)
   - Runtime overrides (highest priority)

2. **Environment Variables**:
   - Use environment variables for sensitive information
   - Document all required environment variables
   - Provide default values where appropriate

3. **Type Safety**:
   - Use Pydantic models for complex configurations
   - Validate configuration on load
   - Provide meaningful error messages for invalid configurations

## Integration with Other Modules

The settings module integrates with:
- `ConfigManager` for configuration loading and merging
- `ResourceManager` for resource directory management
- Model training and evaluation pipelines
- Data loading and preprocessing components

## Example: Custom Settings Class

```python
from pydantic import BaseModel
from culicidaelab.core.settings import Settings as BaseSettings

class ModelConfig(BaseModel):
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100

class CustomSettings(BaseSettings):
    def __init__(self, config_dir=None):
        super().__init__(config_dir)
        self.model_config = self._load_model_config()
    
    def _load_model_config(self) -> ModelConfig:
        """Load and validate model configuration."""
        model_config = self.config.get('model', {})
        return ModelConfig(**model_config)
    
    def get_training_params(self) -> dict:
        """Get training parameters with validation."""
        return self.model_config.dict()

# Usage
settings = CustomSettings()
training_params = settings.get_training_params()
```

## Performance Considerations

1. **Lazy Loading**:
   - Configurations are loaded on first access
   - Environment variables are loaded at startup

2. **Caching**:
   - Settings are cached after first access
   - Resource directories are created on demand

3. **Validation**:
   - Configuration validation happens during initialization
   - Invalid configurations raise descriptive errors

## Troubleshooting

### Common Issues

1. **Configuration Not Found**
   - Verify the configuration directory structure
   - Check file permissions
   - Ensure required files exist

2. **Environment Variables Not Loading**
   - Check variable names and values
   - Ensure `.env` file is in the correct location
   - Restart the application after changing environment variables

3. **Permission Issues**
   - Ensure the application has write access to required directories
   - Check directory permissions

## Contributing

When modifying the settings module:
1. Maintain backward compatibility
2. Add type hints and docstrings
3. Include unit tests
4. Update documentation for new features

---
*Documentation generated on: 2025-05-28*
