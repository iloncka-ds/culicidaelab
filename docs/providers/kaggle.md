# Kaggle Provider

## Overview
The `KaggleProvider` class provides a simple interface for downloading datasets from Kaggle. It wraps the Kaggle API to handle authentication and dataset downloads.

## Features
- **Dataset Downloading**: Download datasets from Kaggle competitions and datasets
- **Authentication**: Secure API key management
- **Simple Interface**: Easy-to-use methods for common operations

## Installation
```bash
pip install culicidaelab kaggle
```

## Configuration

### Environment Variables
```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### Config File
```yaml
providers:
  kaggle:
    username: "your_username"
    api_key: "your_api_key"
```

## Usage

### Initialization
```python
from culicidaelab.providers.kaggle_provider import KaggleProvider

# Create provider instance
provider = KaggleProvider()
```

### Downloading Datasets
```python
# Download a dataset
provider.download_dataset("username/dataset-name", path="./data")

# Download from a competition
provider.download_dataset("c/competition-name", path="./competition-data")
```

## API Reference

### `KaggleProvider` Class

#### Constructor
```python
KaggleProvider()
```
Initializes the Kaggle provider and authenticates using environment variables.

#### Methods

##### `download_dataset(dataset: str, path: str = "./data")`
Download a dataset from Kaggle.

**Parameters**:
- `dataset`: Dataset identifier in format "username/dataset" or "c/competition"
- `path`: Local directory to save the dataset (default: "./data")

## Advanced Usage

### Using with Kaggle API Directly
```python
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize API
api = KaggleApi()
api.authenticate()

# List available datasets
datasets = api.dataset_list()
for dataset in datasets:
    print(dataset)
```

### Downloading Specific Files
```python
# Download only specific files from a dataset
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    'username/dataset-name',
    path='./data',
    unzip=True,
    quiet=False
)
```

## Error Handling

### Handling Authentication Errors
```python
try:
    provider = KaggleProvider()
except ValueError as e:
    print(f"Authentication failed: {e}")
    print("Please set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
```

### Handling Download Failures
```python
try:
    provider.download_dataset("nonexistent/dataset")
except Exception as e:
    print(f"Failed to download dataset: {e}")
```

## Best Practices

1. **API Key Management**
   - Store API keys in environment variables
   - Never commit kaggle.json or API keys to version control
   - Use .gitignore to exclude downloaded datasets

2. **Rate Limiting**
   - Be mindful of Kaggle's rate limits
   - Implement retries with exponential backoff for failed requests

3. **Data Organization**
   - Organize downloaded datasets in a consistent directory structure
   - Include version information in directory names

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   - Verify the API key is valid and has the correct permissions
   - Check that the kaggle.json file is in the correct location (~/.kaggle/)
   - Ensure the API key hasn't been revoked on Kaggle

2. **Dataset Not Found**
   - Verify the dataset name and owner
   - Check if the dataset is private (requires authentication)
   - Ensure you have accepted the competition rules if downloading competition data

3. **Connection Issues**
   - Check your internet connection
   - Verify you can access https://www.kaggle.com
   - Check for any firewall or proxy restrictions

## Integration

The KaggleProvider integrates with:
- Data processing pipelines
- Experiment tracking systems
- Model training workflows

## Example Workflow

```python
from culicidaelab.providers.kaggle_provider import KaggleProvider
import pandas as pd
import os

# Initialize provider
provider = KaggleProvider()

# Download dataset
os.makedirs("data/mosquito-detection", exist_ok=True)
provider.download_dataset("competitions/ultra-mnist", path="data/mosquito-detection")

# Load the dataset
train_df = pd.read_csv("data/mosquito-detection/train.csv")
print(f"Loaded {len(train_df)} training samples")
```
