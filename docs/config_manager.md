# Configuration Manager Module

## Overview
The `config_manager` module provides a robust and flexible configuration management system for the culicidaelab library. It implements the singleton pattern to ensure consistent configuration access throughout the application, handling YAML configurations, environment variables, and dynamic object instantiation.

## Features
- **Singleton Pattern**: Ensures a single source of truth for configuration
- **Multi-source Configuration**: Load and merge configurations from multiple YAML files
- **Environment Variable Support**: Seamless integration with `.env` files
- **Dynamic Object Instantiation**: Create objects directly from configuration
- **Provider Management**: Handle API keys and configurations for external services
- **Resource Management**: Standardized access to application resources

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from culicidaelab.core import ConfigManager

# Initialize the configuration manager
config_manager = ConfigManager()

# Load configuration
config = config_manager.load_config("my_config")

# Access configuration values
value = config_manager.get_config("path.to.setting")
```

### Configuration Loading
```python
# Load with overrides
config = config_manager.load_config(
    "model",
    overrides={"training.epochs": 50, "learning_rate": 0.001}
)
```

### Object Instantiation
```yaml
# config.yaml
data_loader:
  _target_: my_package.DataLoader
  batch_size: 32
  shuffle: true
```

```python
# Python code
data_loader = config_manager.instantiate_from_config("data_loader")
```

## API Reference

### ConfigManager

#### Methods

##### `__init__(self, library_config_path=None, config_path=None, **kwargs)`
Initialize the ConfigManager instance.

**Parameters**:
- `library_config_path` (str, optional): Path to library configuration directory
- `config_path` (str, optional): Path to user configuration directory

##### `load_config(self, config_name="config", overrides=None)`
Load and merge configurations.

**Parameters**:
- `config_name` (str): Name of the configuration file (without extension)
- `overrides` (dict, optional): Dictionary of configuration overrides

**Returns**:
- `DictConfig`: Merged configuration object

##### `get_config(self, config_path=None)`
Retrieve a configuration value or the entire configuration.

**Parameters**:
- `config_path` (str, optional): Dot-separated path to configuration value

**Returns**:
- Any: Configuration value or entire configuration

##### `instantiate_from_config(self, config_path, _target_key="_target_", **kwargs)`
Instantiate an object from configuration.

**Parameters**:
- `config_path` (str): Dot-separated path to configuration
- `_target_key` (str): Key containing the target class path
- `**kwargs`: Additional constructor arguments

**Returns**:
- Any: Instantiated object

##### `get_provider_config(self, provider)`
Get configuration for a specific provider.

**Parameters**:
- `provider` (str): Name of the provider

**Returns**:
- dict: Provider configuration with API key

##### `get_resource_dirs(self)`
Get standard resource directories.

**Returns**:
- dict: Dictionary of resource directories

### ConfigurableComponent
Base class for components that require configuration.

#### Methods

##### `__init__(self, config_manager)`
Initialize the configurable component.

**Parameters**:
- `config_manager` (ConfigManager): Configuration manager instance

##### `load_config(self, config_path=None)`
Load component-specific configuration.

**Parameters**:
- `config_path` (str, optional): Path to configuration

## Configuration Structure

### Default Configuration Paths
1. Library configuration: `{package_root}/conf/`
2. User configuration: `~/.culicidaelab/`
3. Environment variables: Loaded from `.env` file if present

### Environment Variables
- `KAGGLE_API_KEY`: API key for Kaggle
- `HUGGINGFACE_API_KEY`: API key for Hugging Face
- `ROBOFLOW_API_KEY`: API key for Roboflow

## Examples

### Custom Configuration Path
```python
config_manager = ConfigManager(
    library_config_path="/path/to/library/config",
    config_path="/path/to/user/config"
)
```

### Using ConfigurableComponent
```python
class MyComponent(ConfigurableComponent):
    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.load_config()
        
    def process(self):
        # Access configuration
        param = self.config["my_parameter"]
        # ...
```

## Best Practices
1. **Configuration Hierarchy**:
   - Library defaults (lowest priority)
   - User configuration (medium priority)
   - Runtime overrides (highest priority)

2. **Error Handling**:
   - Always check if configuration is loaded before access
   - Use try-except blocks for critical configuration sections

3. **Security**:
   - Never commit sensitive data to version control
   - Use environment variables for secrets

## Troubleshooting

### Common Issues
1. **Configuration Not Found**
   - Verify the configuration file exists in the expected location
   - Check file permissions

2. **Environment Variables Not Loading**
   - Ensure `.env` file is in the project root
   - Verify variable names match exactly

## Contributing
Contributions are welcome! Please follow the project's coding standards and submit a pull request.

## License
[Specify License]

---
*Documentation generated on: 2025-05-28*
