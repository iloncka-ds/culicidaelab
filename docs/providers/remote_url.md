# Remote URL Provider

## Overview
The `RemoteURLProvider` class provides a flexible way to download files from any public URL. It's designed to handle various file types and includes features like progress tracking and resumable downloads.

## Features
- **Universal Downloading**: Download files from any public URL
- **Resumable Downloads**: Continue interrupted downloads
- **Progress Tracking**: Monitor download progress
- **Flexible Storage**: Save to custom locations
- **Metadata Retrieval**: Get file information before downloading

## Installation
```bash
pip install culicidaelab requests
```

## Configuration

### Basic Configuration
```yaml
providers:
  remote_url:
    downloads_dir: "./downloads"  # Default download directory
    timeout: 30  # Request timeout in seconds
    max_retries: 3  # Number of retry attempts
```

## Usage

### Initialization
```python
from culicidaelab.core import ConfigManager
from culicidaelab.providers.remote_url_provider import RemoteURLProvider

# Initialize configuration
config_manager = ConfigManager()

# Create provider instance
provider = RemoteURLProvider(config_manager)
```

### Downloading Files
```python
# Basic download
file_path = provider.download_dataset(
    "https://example.com/dataset.zip",
    destination="./data/dataset.zip"
)

# Download to default downloads directory
file_path = provider.download_dataset("https://example.com/data.csv")
```

### Getting File Metadata
```python
# Get file metadata without downloading
metadata = provider.get_metadata("https://example.com/large_file.zip")
print(f"File size: {metadata.get('content_length') / (1024*1024):.2f} MB")
print(f"Content type: {metadata.get('content_type')}")
```

## API Reference

### `RemoteURLProvider` Class

#### Constructor
```python
RemoteURLProvider(config_manager: ConfigManager)
```

**Parameters**:
- `config_manager`: Instance of `ConfigManager` with provider configuration

#### Methods

##### `download_dataset(url: str, destination: str | Path | None = None, **kwargs) -> Path`
Download a file from a remote URL.

**Parameters**:
- `url`: The URL of the file to download
- `destination`: Optional local path to save the file
- `**kwargs`: Additional download parameters

**Returns**:
- Path to the downloaded file

##### `get_metadata(url: str) -> dict[str, Any]`
Retrieve metadata about a remote file.

**Parameters**:
- `url`: The URL of the remote file

**Returns**:
- Dictionary containing file metadata

## Advanced Usage

### Custom Download Directory
```python
# Set custom download directory in config
config = config_manager.get_config()
config.providers.remote_url.downloads_dir = "./custom_downloads"
config_manager.update_config(config)

# Now downloads will go to ./custom_downloads
file_path = provider.download_dataset("https://example.com/data.csv")
```

### Download with Progress Bar
```python
from tqdm import tqdm

def download_with_progress(url, destination):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(destination, 'wb') as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

# Use the custom downloader
provider.download_dataset = download_with_progress
provider.download_dataset("https://example.com/large_file.zip", "large_file.zip")
```

## Error Handling

### Handling Download Errors
```python
try:
    file_path = provider.download_dataset("https://example.com/nonexistent.file")
except Exception as e:
    print(f"Download failed: {e}")
```

### Handling Network Issues
```python
import time
from requests.exceptions import RequestException

max_retries = 3
retry_delay = 5  # seconds

for attempt in range(max_retries):
    try:
        file_path = provider.download_dataset("https://example.com/unstable.file")
        break  # Success, exit the retry loop
    except RequestException as e:
        if attempt == max_retries - 1:
            print(f"Failed after {max_retries} attempts: {e}")
            raise
        print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
```

## Best Practices

1. **Error Handling**
   - Always wrap downloads in try-except blocks
   - Implement retries for transient failures
   - Validate downloaded files (e.g., checksums)

2. **Performance**
   - Use streaming for large files
   - Set appropriate timeouts
   - Consider parallel downloads for multiple files

3. **Security**
   - Verify SSL certificates
   - Sanitize filenames from URLs
   - Set download size limits

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**
   - Update your certificate store
   - Use `verify=False` only for testing (not recommended for production)

2. **Slow Downloads**
   - Check your network connection
   - Try a different network or use a CDN if available
   - Consider using a download manager

3. **Partial Downloads**
   - Implement resumable downloads
   - Verify file integrity after download
   - Use the `Range` header for partial content

## Integration

The RemoteURLProvider integrates with:
- Data processing pipelines
- File management systems
- Caching mechanisms
- Progress tracking systems
