# Resource Manager Module

## Overview
The `resource_manager` module provides a robust, cross-platform solution for managing application resources including models, datasets, cache files, and temporary workspaces. It implements a thread-safe, singleton pattern to ensure consistent resource access throughout the application.

## Features
- **Cross-Platform Compatibility**: Works consistently across Windows, macOS, and Linux
- **Thread-Safe Operations**: Safe for use in multi-threaded applications
- **Comprehensive Path Management**: Standardized paths for different resource types
- **Temporary Workspace Management**: Context managers for temporary workspaces
- **File Operations**: Utilities for safe file operations with integrity checking
- **Automatic Cleanup**: Configurable cleanup of temporary resources

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from culicidaelab.core import ResourceManager

# Initialize the resource manager
resource_manager = ResourceManager(app_name="my_app")

# Get standard resource directories
model_dir = resource_manager.model_dir
dataset_dir = resource_manager.dataset_dir
```

### Using Temporary Workspaces
```python
with resource_manager.temp_workspace("processing_job") as workspace:
    # Work with files in the temporary workspace
    output_file = workspace / "result.txt"
    output_file.write_text("Processing complete")

    # Workspace will be automatically cleaned up when the context exits
```

## API Reference

### ResourceManager

#### Constructor
```python
ResourceManager(app_name: str | None = None, custom_base_dir: str | Path | None = None)
```

**Parameters**:
- `app_name` (str, optional): Custom application name. If None, attempts to load from pyproject.toml.
- `custom_base_dir` (str | Path, optional): Custom base directory for all resources. If None, uses system defaults.

#### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `app_name` | str | Name of the application |
| `user_data_dir` | Path | User-specific data directory |
| `user_cache_dir` | Path | User-specific cache directory |
| `temp_dir` | Path | Temporary directory for runtime operations |
| `model_dir` | Path | Directory for model files |
| `dataset_dir` | Path | Directory for dataset files |
| `downloads_dir` | Path | Directory for downloaded files |
| `logs_dir` | Path | Directory for log files |
| `config_dir` | Path | Directory for configuration files |

#### Methods

##### `get_all_directories() -> dict[str, Path]`
Get a dictionary of all managed directories.

**Returns**:
- `dict[str, Path]`: Mapping of directory names to their paths

##### `temp_workspace(prefix: str = "", cleanup: bool = True) -> ContextManager[Path]`
Create a temporary workspace directory.

**Parameters**:
- `prefix` (str): Prefix for the workspace directory name
- `cleanup` (bool): Whether to clean up the workspace on exit

**Returns**:
- `ContextManager[Path]`: Context manager yielding the workspace path

##### `get_file_path(relative_path: str, create_parents: bool = True) -> Path`
Get an absolute path relative to the data directory.

**Parameters**:
- `relative_path` (str): Relative path within the data directory
- `create_parents` (bool): Whether to create parent directories

**Returns**:
- `Path`: Absolute path to the requested file

##### `clean_temp_dirs(older_than_hours: int = 24) -> None`
Clean up temporary directories older than specified hours.

**Parameters**:
- `older_than_hours` (int): Age threshold in hours for cleanup

## Advanced Usage

### Custom Base Directory
```python
# Use a custom base directory for all resources
resource_manager = ResourceManager(
    app_name="my_app",
    custom_base_dir="/custom/path"
)
```

### File Operations
```python
# Safe file operations
source = Path("local/file.txt")
destination = resource_manager.get_file_path("processed/file.txt")

# Safe copy with overwrite protection
resource_manager.safe_copy(source, destination)


# Calculate file hash
file_hash = resource_manager.calculate_file_hash(source)
```

### Workspace Management
```python
# Create a named workspace that persists
workspace = resource_manager.create_workspace("training_run")
try:
    # Use the workspace
    model_file = workspace / "model.pt"
    # ...
finally:
    # Clean up when done
    resource_manager.cleanup_workspace("training_run")
```

## Best Practices

1. **Resource Isolation**: Use separate workspaces for different processing tasks
2. **Cleanup**: Always use context managers (`with` statements) for temporary resources
3. **Path Handling**: Always use the ResourceManager to get paths instead of hardcoding them
4. **Error Handling**: Handle `ResourceManagerError` for filesystem operations
5. **Thread Safety**: The ResourceManager is thread-safe, but individual file operations may require additional synchronization

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**
   - Ensure the application has write permissions to the specified directories
   - On Unix-like systems, check directory permissions (should be 755 for directories)

2. **Directory Not Found**
   - Verify that the base directory exists and is accessible
   - Check for typos in the application name or custom paths

3. **Temporary Files Not Cleaning Up**
   - Ensure `cleanup=True` when using context managers
   - Call `clean_temp_dirs()` periodically for long-running applications

## Integration with Other Modules

The ResourceManager integrates with other modules in the library:
- Used by `ConfigManager` for configuration file management
- Integrated with dataset loaders for managing dataset storage
- Works with model predictors for model file management

## Performance Considerations

- **Caching**: Directory paths are cached after first access
- **Lazy Initialization**: Resources are only created when first accessed
- **Cleaning**: Regular cleanup of temporary files prevents disk space issues

## Contributing

When adding new resource types or features:
1. Follow the existing directory structure
2. Add appropriate cleanup mechanisms
3. Ensure thread safety for all public methods
4. Add relevant tests in `tests/core/test_resource_manager.py`

---
*Documentation generated on: 2025-05-28*
