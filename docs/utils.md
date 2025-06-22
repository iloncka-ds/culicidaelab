# Utilities Module

## Overview
The `utils` module provides essential utility functions that are used throughout the culicidaelab library. These utilities include file downloading with progress tracking, path handling, and other common operations.

## Features
- **File Downloading**: Download files from URLs with progress tracking
- **Progress Reporting**: Built-in support for progress bars and callbacks
- **Error Handling**: Comprehensive error handling and reporting
- **Cross-Platform**: Works consistently across different operating systems

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Downloading Files
```python
from culicidaelab.core.utils import download_file, default_progress_callback

# Download a file with default settings
file_path = download_file("https://example.com/data.zip")

# Download with custom destination and progress callback
download_file(
    url="https://example.com/model.pt",
    destination="models/pretrained.pt",
    progress_callback=default_progress_callback,
    desc="Downloading model weights"
)
```

## API Reference

### `download_file` Function

```python
download_file(
    url: str,
    destination: str | Path | None = None,
    downloads_dir: str | Path | None = None,
    progress_callback: Callable | None = None,
    chunk_size: int = 8192,
    timeout: int = 30,
    desc: str | None = None
) -> Path
```

Download a file from a URL with progress tracking.

**Parameters**:
- `url` (str): URL of the file to download
- `destination` (str | Path, optional): Specific destination path for the file
- `downloads_dir` (str | Path, optional): Default directory for downloads if no destination specified
- `progress_callback` (Callable, optional): Function to track download progress
- `chunk_size` (int): Size of chunks to download (default: 8192 bytes)
- `timeout` (int): Timeout for the download request in seconds (default: 30)
- `desc` (str, optional): Description for the progress bar

**Returns**:
- `Path`: Path to the downloaded file

**Raises**:
- `ValueError`: If URL is invalid
- `RuntimeError`: If download fails

### `default_progress_callback` Function

```python
default_progress_callback(downloaded: int, total: int) -> None
```

Default progress callback that prints download progress.

**Parameters**:
- `downloaded` (int): Number of bytes downloaded
- `total` (int): Total number of bytes to download

## Advanced Usage

### Custom Progress Callback
```python
def custom_progress(downloaded: int, total: int) -> None:
    """Custom progress callback that shows a simple progress bar."""
    bar_length = 50
    if total > 0:
        progress = downloaded / total
        bar = '=' * int(bar_length * progress)
        print(f'\r[{bar.ljust(bar_length)}] {progress:.1%}', end='', flush=True)
        if downloaded >= total:
            print()  # New line when complete

# Usage
download_file(
    url="https://example.com/large_file.zip",
    progress_callback=custom_progress
)
```

### Using with ResourceManager
```python
from culicidaelab.core import ResourceManager
from culicidaelab.core.utils import download_file

# Initialize resource manager
resource_manager = ResourceManager()

# Download to a resource directory
model_url = "https://example.com/pretrained_model.pt"
model_path = resource_manager.model_dir / "pretrained.pt"

download_file(
    url=model_url,
    destination=model_path,
    desc="Downloading model weights"
)
```

## Error Handling

The `download_file` function raises specific exceptions for different error scenarios:

1. **Invalid URL**:
   ```python
   try:
       download_file("invalid-url")
   except ValueError as e:
       print(f"Invalid URL: {e}")
   ```

2. **Download Failures**:
   ```python
   try:
       download_file("https://example.com/nonexistent.file")
   except RuntimeError as e:
       print(f"Download failed: {e}")
   ```

3. **File System Errors**:
   ```python
   try:
       download_file("https://example.com/file.txt", "/invalid/path/file.txt")
   except RuntimeError as e:
       print(f"File system error: {e}")
   ```

## Performance Considerations

1. **Chunk Size**:
   - Larger chunk sizes reduce the number of write operations but increase memory usage
   - Default of 8KB provides a good balance for most use cases
   - Adjust based on your specific requirements and system resources

2. **Progress Updates**:
   - Progress callbacks are called for each chunk
   - For very large files, consider throttling updates to reduce overhead

3. **Timeouts**:
   - The default timeout is 30 seconds
   - Increase for large files or slow connections
   - Consider implementing retry logic for unreliable connections

## Example: Downloading Model Weights

```python
from pathlib import Path
from culicidaelab.core.utils import download_file

def download_model_weights(
    model_url: str,
    model_name: str,
    models_dir: Path = Path("models")
) -> Path:
    """
    Download model weights from a URL.

    Args:
        model_url: URL of the model weights
        model_name: Name for the downloaded file
        models_dir: Directory to save the model

    Returns:
        Path to the downloaded model weights
    """
    # Create models directory if it doesn't exist
    models_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension from URL or use .pth as default
    file_extension = Path(model_url).suffix or ".pth"
    destination = models_dir / f"{model_name}{file_extension}"

    # Download with progress
    print(f"Downloading {model_name} model...")
    download_file(
        url=model_url,
        destination=destination,
        desc=f"Downloading {model_name}"
    )

    print(f"Model saved to {destination}")
    return destination

# Usage
model_url = "https://example.com/pretrained_models/mosquito_classifier_v1.0.0.pth"
download_model_weights(model_url, "mosquito_classifier")
```

## Best Practices

1. **Error Handling**:
   - Always wrap download operations in try-except blocks
   - Provide meaningful error messages to users
   - Clean up partially downloaded files on failure

2. **Progress Feedback**:
   - Use progress bars for downloads expected to take more than a few seconds
   - Provide estimated time remaining when possible

3. **Resource Management**:
   - Use context managers for file operations
   - Close network connections properly
   - Handle timeouts gracefully

## Troubleshooting

### Common Issues

1. **Download Stalls**
   - Check network connectivity
   - Increase the timeout value
   - Verify the server supports resumable downloads

2. **Permission Errors**
   - Ensure the destination directory is writable
   - Check for sufficient disk space

3. **SSL Certificate Errors**
   - Verify the server's SSL certificate
   - Use `verify=False` in development (not recommended for production)

## Contributing

When adding new utility functions:
1. Add type hints for all parameters and return values
2. Include comprehensive docstrings
3. Add unit tests
4. Document any dependencies
5. Follow the project's coding style

---
*Documentation generated on: 2025-05-28*
