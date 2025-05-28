# Mosquito Detector

## Overview
The `MosquitoDetector` class provides functionality for detecting mosquitoes in images using YOLO (You Only Look Once) object detection models. It's built on top of the Ultralytics YOLO implementation and integrates with the culicidaelab ecosystem.

## Features
- **YOLO Integration**: Supports various YOLO model versions (v5, v8, etc.)
- **Object Detection**: Detects and localizes mosquitoes in images
- **Batch Processing**: Efficient processing of multiple images
- **Visualization**: Built-in visualization of detection results
- **Evaluation**: Comprehensive object detection metrics (mAP, IoU, etc.)
- **GPU Acceleration**: Utilizes GPU when available for faster inference

## Installation
```bash
pip install culicidaelab ultralytics opencv-python
```

## Quick Start

### Basic Usage
```python
from culicidaelab.predictors.detector import MosquitoDetector
from culicidaelab.core import ConfigManager
import cv2

# Initialize configuration manager
config_manager = ConfigManager()

# Initialize detector
detector = MosquitoDetector(
    model_path="path/to/yolo_model.pt",
    config_manager=config_manager
)

# Load an image (BGR format as returned by OpenCV)
image = cv2.imread("mosquito_image.jpg")

# Detect mosquitoes
detections = detector.predict(image)

# Visualize results
visualization = detector.visualize(image, detections)
cv2.imshow("Detections", visualization)
cv2.waitKey(0)
```

## API Reference

### `MosquitoDetector` Class

#### Constructor
```python
MosquitoDetector(
    model_path: str | Path,
    config_manager: ConfigManager
)
```

**Parameters**:
- `model_path`: Path to pre-trained YOLO model weights (.pt file)
- `config_manager`: Instance of `ConfigManager` containing detector settings

#### Methods

##### `predict(input_data: np.ndarray) -> list[tuple[float, float, float, float, float]]`
Detect mosquitoes in an image.

**Parameters**:
- `input_data`: Input image as numpy array in BGR format (H, W, 3)

**Returns**:
- List of detections, where each detection is a tuple (center_x, center_y, width, height, confidence)

##### `evaluate(ground_truth: list[dict], predictions: list[list[tuple]] = None) -> dict`
Evaluate detection performance.

**Parameters**:
- `ground_truth`: List of ground truth annotations
- `predictions`: Optional pre-computed predictions

**Returns**:
- Dictionary of evaluation metrics (mAP, precision, recall, etc.)

##### `visualize(image: np.ndarray, detections: list[tuple], save_path: str = None) -> np.ndarray`
Visualize detection results on the input image.

**Parameters**:
- `image`: Input image as numpy array
- `detections`: List of detections from predict()
- `save_path`: Optional path to save the visualization

**Returns**:
- Image with detection visualizations

## Configuration

The detector requires a configuration with the following structure:

```yaml
detector:
  model:
    confidence_threshold: 0.25  # Minimum confidence score for detections
    iou_threshold: 0.45         # IOU threshold for NMS
    max_detections: 100         # Maximum number of detections per image
    device: "cuda:0"            # Device to run inference on (cuda:0, cpu, etc.)
  
  visualization:
    box_color: [0, 255, 0]     # BGR color for bounding boxes
    box_thickness: 2            # Thickness of bounding box lines
    text_color: [255, 255, 255] # BGR color for text
    text_thickness: 1           # Thickness of text
    font_scale: 0.5             # Font scale for text
```

## Advanced Usage

### Batch Processing
```python
# List of image paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Process all images
detection_results = []
for path in image_paths:
    image = cv2.imread(path)
    detections = detector.predict(image)
    detection_results.append(detections)
    
    # Visualize and save
    vis = detector.visualize(image, detections)
    cv2.imwrite(f"detected_{path}", vis)
```

### Customizing Visualization
```python
# Get detections
detections = detector.predict(image)

# Custom visualization
for x, y, w, h, conf in detections:
    # Convert center coordinates to corner coordinates
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    
    # Draw custom bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
    
    # Add custom label
    label = f"Mosquito: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
```

## Best Practices

1. **Model Selection**:
   - Choose appropriate YOLO model size based on speed/accuracy requirements
   - Larger models (YOLOv8x) offer better accuracy but are slower
   - Smaller models (YOLOv8n) are faster but may have lower accuracy

2. **Performance Optimization**:
   - Use batch processing for multiple images
   - Enable GPU acceleration when available
   - Consider model quantization for edge deployment

3. **Post-Processing**:
   - Adjust confidence threshold based on your application needs
   - Use appropriate NMS (Non-Maximum Suppression) settings

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify the model file exists and is a valid YOLO model
   - Check for version compatibility between model and ultralytics package
   - Ensure CUDA is properly installed if using GPU

2. **Low Detection Accuracy**
   - Adjust confidence and NMS thresholds
   - Consider fine-tuning the model on your specific data
   - Check for class imbalance in your training data

3. **Performance Issues**
   - Reduce input image size for faster inference
   - Use a smaller model if real-time performance is critical
   - Enable half-precision inference if supported by your hardware

## Integration with Other Components

The `MosquitoDetector` integrates with:
- `ConfigManager` for configuration management
- `ResourceManager` for model loading
- Data preprocessing pipelines
- Visualization tools
- Evaluation frameworks

---
*Documentation generated on: 2025-05-29*
