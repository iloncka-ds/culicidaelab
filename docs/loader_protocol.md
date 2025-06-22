# Loader Protocol Module

## Overview
The `loader_protocol` module defines the protocol for dataset loading strategies in the culicidaelab library. It provides a standardized interface for loading datasets from various sources, ensuring consistency across different data loaders.

## Features
- **Type-Safe Interface**: Uses Python's Protocol for defining the dataset loading contract
- **Flexible Implementation**: Supports loading datasets from different sources and formats
- **Generic Type Support**: Works with any dataset type through Python's generics
- **Simple and Extensible**: Easy to implement for new data sources

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from typing import Any
from culicidaelab.core.loader_protocol import DatasetLoader

# Example implementation of DatasetLoader
class CustomDatasetLoader:
    def load_dataset(self, path: str, split: str | None = None, **kwargs) -> Any:
        """Load a dataset from the given path."""
        # Implementation for loading dataset
        return loaded_dataset

# Usage
data_loader: DatasetLoader = CustomDatasetLoader()
dataset = data_loader.load_dataset("path/to/dataset", split="train")
```

## API Reference

### `DatasetLoader` Protocol

#### `load_dataset(path: str, split: str | None = None, **kwargs) -> T`
Load a dataset from the specified path.

**Parameters**:
- `path` (str): Path to the dataset or dataset identifier
- `split` (str, optional): Dataset split to load (e.g., 'train', 'val', 'test')
- `**kwargs`: Additional arguments specific to the dataset loader implementation

**Returns**:
- `T`: The loaded dataset, type depends on the implementation

## Implementation Guide

### Creating a Custom Dataset Loader

1. **Basic Implementation**:
   ```python
   from typing import Any
   from culicidaelab.core.loader_protocol import DatasetLoader

   class MyDatasetLoader:
       def load_dataset(self, path: str, split: str | None = None, **kwargs) -> Any:
           """
           Load a dataset from the given path.

           Args:
               path: Path to the dataset or dataset identifier
               split: Dataset split to load (e.g., 'train', 'val', 'test')
               **kwargs: Additional arguments

           Returns:
               The loaded dataset
           """
           # Implementation for loading your specific dataset format
           # Example: Load from local files, remote storage, etc.
           return loaded_dataset
   ```

2. **Type-Safe Implementation**:
   ```python
   from typing import Dict, List
   from dataclasses import dataclass
   from culicidaelab.core.loader_protocol import DatasetLoader

   @dataclass
   class DataPoint:
       image: bytes
       label: int
       metadata: Dict[str, Any]

   class TypedDatasetLoader:
       def load_dataset(
           self,
           path: str,
           split: str | None = None,
           **kwargs
       ) -> List[DataPoint]:
           """Load and return a list of typed data points."""
           # Implementation that returns List[DataPoint]
           return data_points
   ```

## Integration with Other Modules

The `DatasetLoader` protocol is designed to work with:
- Data preprocessing pipelines
- Model training loops
- Evaluation frameworks
- Data visualization tools

## Best Practices

1. **Error Handling**:
   - Validate input paths and parameters
   - Provide meaningful error messages for invalid inputs
   - Handle missing or corrupted data gracefully

2. **Performance**:
   - Implement lazy loading for large datasets
   - Support memory-efficient data streaming
   - Cache loaded datasets when appropriate

3. **Documentation**:
   - Document all parameters in the docstring
   - Include examples in the docstring
   - Specify the return type and structure

## Example Implementations

### Image Dataset Loader
```python
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import numpy as np
from culicidaelab.core.loader_protocol import DatasetLoader

class ImageDatasetLoader:
    """Loader for image classification datasets."""

    def __init__(self, image_size: tuple[int, int] = (224, 224)):
        self.image_size = image_size

    def load_dataset(
        self,
        path: str,
        split: str | None = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load images from a directory structure:

        path/
          class1/
            img1.jpg
            img2.jpg
          class2/
            img3.jpg
            ...
        """
        path = Path(path)
        data = []

        # If split is specified, look in a subdirectory
        if split:
            path = path / split

        # Load images from class directories
        for class_dir in path.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            for img_path in class_dir.glob('*.jpg'):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.image_size)

                    data.append({
                        'image': np.array(img),
                        'label': class_name,
                        'path': str(img_path)
                    })
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        return data
```

### CSV Dataset Loader
```python
from typing import List, Dict, Any
import pandas as pd
from culicidaelab.core.loader_protocol import DatasetLoader

class CSVDatasetLoader:
    """Loader for tabular datasets in CSV format."""

    def load_dataset(
        self,
        path: str,
        split: str | None = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load data from a CSV file.

        Args:
            path: Path to the CSV file
            split: Optional split name (e.g., 'train', 'test')
            **kwargs: Additional arguments passed to pandas.read_csv()

        Returns:
            List of dictionaries, one per row in the CSV
        """
        df = pd.read_csv(path, **kwargs)
        return df.to_dict('records')
```

## Performance Considerations

1. **Lazy Loading**:
   - For large datasets, implement lazy loading to avoid loading everything into memory
   - Use Python generators to yield data points one at a time

2. **Caching**:
   - Cache processed datasets to avoid reprocessing
   - Consider using `@functools.lru_cache` for expensive operations

3. **Parallel Loading**:
   - Implement parallel loading for better performance
   - Use `concurrent.futures` or similar for I/O-bound operations

## Contributing

When implementing a new dataset loader:
1. Follow the `DatasetLoader` protocol
2. Add type hints for better IDE support
3. Include comprehensive docstrings
4. Add unit tests
5. Document any additional dependencies

---
*Documentation generated on: 2025-05-28*
