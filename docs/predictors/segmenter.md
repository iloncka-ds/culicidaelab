# Mosquito Segmenter

## Overview
The `MosquitoSegmenter` class provides functionality for segmenting mosquitoes in images using SAM (Segment Anything Model). It's designed to work with both automatic segmentation and segmentation guided by detection boxes.

## Features
- **SAM Integration**: Utilizes the Segment Anything Model for high-quality segmentation
- **Flexible Input**: Works with both automatic and detection-guided segmentation
- **Visualization**: Built-in visualization of segmentation results
- **Evaluation**: Comprehensive metrics for segmentation quality
- **GPU Acceleration**: Leverages GPU when available for faster inference

## Installation
```bash
pip install culicidaelab opencv-python torch torchvision
```

## Quick Start

### Basic Usage
```python
from culicidaelab.predictors.segmenter import MosquitoSegmenter
from culicidaelab.core import ConfigManager
import cv2

# Initialize configuration manager
config_manager = ConfigManager()

# Initialize segmenter
segmenter = MosquitoSegmenter(
    model_path="path/to/sam_model.pth",
    config_manager=config_manager
)

# Load an image (BGR format as returned by OpenCV)
image = cv2.imread("mosquito_image.jpg")

# Perform segmentation
segmentation_mask = segmenter.predict(image)

# Visualize results
visualization = segmenter.visualize(image, segmentation_mask)
cv2.imshow("Segmentation", visualization)
cv2.waitKey(0)
```

## API Reference

### `MosquitoSegmenter` Class

#### Constructor
```python
MosquitoSegmenter(
    model_path: str | Path,
    config_manager: ConfigManager
)
```

**Parameters**:
- `model_path`: Path to SAM model checkpoint file
- `config_manager`: Instance of `ConfigManager` containing segmentation settings

#### Methods

##### `predict(input_data: np.ndarray, detection_boxes: list[tuple] = None) -> np.ndarray`
Segment mosquitoes in an image.

**Parameters**:
- `input_data`: Input image as numpy array (H, W, 3)
- `detection_boxes`: Optional list of detection boxes in format [(x, y, w, h, conf), ...]

**Returns**:
- Binary segmentation mask (H, W) where True indicates mosquito pixels

##### `visualize(input_data: np.ndarray, predictions: np.ndarray, save_path: str = None) -> np.ndarray`
Visualize segmentation results on the input image.

**Parameters**:
- `input_data`: Original input image
- `predictions`: Binary segmentation mask from predict()
- `save_path`: Optional path to save the visualization

**Returns**:
- Image with segmentation overlay

##### `evaluate(input_data: np.ndarray, ground_truth: np.ndarray) -> dict`
Evaluate segmentation quality against ground truth.

**Parameters**:
- `input_data`: Input image
- `ground_truth`: Ground truth binary mask

**Returns**:
- Dictionary containing metrics: iou, precision, recall, f1

## Configuration

The segmenter requires a configuration with the following structure:

```yaml
model:
  sam_config_path: "path/to/sam_config.yaml"

visualization:
  alpha: 0.7  # Opacity of segmentation overlay
  overlay_color: [0, 255, 0]  # BGR color for segmentation overlay
```

## Advanced Usage

### Using with Detection Boxes
```python
# Assume we have detection boxes from a detector
detection_boxes = [
    (100, 100, 50, 50, 0.95),  # (x, y, w, h, confidence)
    (200, 150, 60, 60, 0.92)
]

# Perform segmentation using detection boxes as prompts
segmentation_mask = segmenter.predict(image, detection_boxes=detection_boxes)
```

### Batch Processing
```python
# List of image paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Process all images
for path in image_paths:
    image = cv2.imread(path)
    
    # Get segmentation mask
    mask = segmenter.predict(image)
    
    # Visualize and save
    vis = segmenter.visualize(image, mask)
    cv2.imwrite(f"segmented_{path}", vis)
```

## Best Practices

1. **Input Preparation**:
   - Ensure images are in the correct color space (BGR for OpenCV, RGB for some models)
   - Normalize pixel values if required by the model
   - Resize images to optimal dimensions if needed

2. **Performance Optimization**:
   - Use detection boxes when available for more precise segmentation
   - Process multiple images in batches if possible
   - Enable GPU acceleration for larger images

3. **Post-Processing**:
   - Apply morphological operations to clean up segmentation masks
   - Consider using connected components to handle multiple instances
   - Apply confidence thresholds to filter out low-quality segments

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce input image size
   - Use smaller batch sizes
   - Enable mixed precision training if available

2. **Poor Segmentation Quality**
   - Verify input image quality and resolution
   - Check if detection boxes are accurate
   - Consider fine-tuning the SAM model on mosquito-specific data

3. **Slow Performance**
   - Enable GPU acceleration if available
   - Reduce input image size
   - Use a smaller SAM model variant

## Integration with Other Components

The `MosquitoSegmenter` integrates with:
- `ConfigManager` for configuration management
- Detection models for guided segmentation
- Visualization tools for result analysis
- Evaluation frameworks for performance measurement

---
*Documentation generated on: 2025-05-29*
