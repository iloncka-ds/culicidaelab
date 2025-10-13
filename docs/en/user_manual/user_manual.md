
# CulicidaeLab User Guide

Welcome to the `CulicidaeLab` User Guide! This guide will walk you through the essential features of the library, showing you how to perform mosquito detection, classification, and segmentation. We will cover everything from initial setup to running predictions and visualizing the results.

## 1. Before You Begin: Core Concepts

### The `settings` Object

The single most important concept in `CulicidaeLab` is the **`settings` object**. It is your centralized entry point for everything. Instead of importing and initializing each component manually with complex parameters, you simply:

1. Get the `settings` object.
2. Ask it to create the component you need (`Detector`, `Classifier`, etc.).

The `settings` object automatically handles loading configurations, managing file paths, downloading models, and selecting the optimal backend for your use case.

### The Predictor Workflow

All three main components (`MosquitoDetector`, `MosquitoClassifier`, `MosquitoSegmenter`) are **Predictors**. They share a consistent and predictable workflow:

1. **Initialize**: Create an instance using the simplified constructor that automatically selects the best backend.
2. **Load Model**: The model weights are lazy-loaded, meaning they are only downloaded and loaded into memory on the first prediction or when you explicitly tell them to. We will use `load_model=True` for clarity.
3. **Predict**: Use the `.predict()` method on an image to get structured, type-safe results.
4. **Visualize**: Use the `.visualize()` method to see the results.

### Structured Prediction Outputs

CulicidaeLab now returns **structured prediction results** instead of simple tuples or lists. This means:

- **Type Safety**: All outputs are validated Pydantic models with clear structure
- **Easy Access**: Use convenient methods like `.top_prediction()` for classification results
- **Rich Information**: Each prediction includes confidence scores, bounding boxes, and metadata
- **JSON Serializable**: Perfect for web APIs and data storage

### Automatic Backend Selection

The library automatically chooses the best backend for your needs:

- **Development Mode**: Uses PyTorch for full flexibility and debugging capabilities
- **Production Mode**: Automatically selects ONNX for optimal performance and smaller memory footprint
- **Transparent**: The same prediction API works regardless of the backend

Let's get started!

## 2. Initialization and Setup

First, let's set up our environment and get the all-important `settings` object.

```python
# Import necessary libraries for image handling and plotting
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Import the main CulicidaeLab components
from culicidaelab import get_settings, MosquitoDetector, MosquitoClassifier, MosquitoSegmenter

# Get the central settings object.
# This single object will be used to initialize all other components.
settings = get_settings()

print("CulicidaeLab settings initialized successfully.")
```

## 3. Mosquito Detection

The first step in many analysis pipelines is to find out *if* a mosquito is in the image and *where* it is. This is the job of the `MosquitoDetector`.

### 3.1. Initializing the Detector and Loading an Image

When we initialize the detector with `load_model=True`, the library checks if the YOLO model weights are present locally. If not, they will be automatically downloaded and cached for all future uses.

```python
# Initialize the detector.
# With load_model=True, the model weights will be loaded into memory.
print("Initializing MosquitoDetector...")
detector = MosquitoDetector(settings=settings, load_model=True)
print("Detector model loaded and ready.")

# Let's load a test image to work with.
# Make sure to replace this path with the path to your own image.
image_path = Path("test_imgs") / "640px-Aedes_aegypti.jpg"

# CulicidaeLab accepts various image input formats:
# - File path (str or Path)
# - PIL Image (already in RGB)
# - NumPy array
# - Image bytes
# For simplicity, we use PIL Image which is already in RGB mode
image = Image.open(image_path)
image_rgb = np.array(image)
```

### 3.2. Performing Detection

The `predict` method takes an image as input and returns a structured `DetectionPrediction` object containing all detected mosquitoes. Each detection includes a bounding box, confidence score, and additional metadata in a type-safe format.

```python
# Run the prediction on our RGB image
result = detector.predict(image_rgb)

# Let's print the results in a human-readable format.
print("\nDetection Results:")
if result.detections:
    for i, detection in enumerate(result.detections):
        bbox = detection.box
        conf = detection.confidence
        print(
            f"  - Mosquito {i+1}: Confidence = {conf:.2f}, "
            f"Box = (x1={bbox.x1:.1f}, y1={bbox.y1:.1f}, "
            f"x2={bbox.x2:.1f}, y2={bbox.y2:.1f})"
        )
else:
    print("  No mosquitoes were detected in the image.")
```

### 3.3. Visualizing Detection Results

Reading coordinates is useful, but seeing the result is better. The `.visualize()` method draws the bounding boxes and confidence scores directly onto the image.

```python
# Pass the original image and the detection result to the visualize method
annotated_image = detector.visualize(image_rgb, result)

# Use matplotlib to display the final image
plt.figure(figsize=(10, 7))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Detected Mosquitoes")
plt.show()
```

## 4. Mosquito Species Classification

Once you have an image of a mosquito, the next question is often, "What species is it?" The `MosquitoClassifier` is trained to answer this.

### 4.1. Initializing the Classifier

Similar to the detector, we initialize the classifier from the `settings` object. The corresponding classification model will be downloaded on first use.

```python
# Initialize the classifier
print("\nInitializing MosquitoClassifier...")
classifier = MosquitoClassifier(settings=settings, load_model=True)
print("Classifier model loaded and ready.")

# For this example, we'll use the same image we used for detection.
# In a real application, you might use a cropped image from the detector's output.
```

### 4.2. Performing Classification

The classifier's `predict` method returns a structured `ClassificationPrediction` object containing all possible species predictions, sorted by confidence score. You can easily access the top prediction or iterate through all predictions.

```python
# Run the classification
result = classifier.predict(image_rgb)

# Print the top prediction using the convenient method
top_pred = result.top_prediction()
if top_pred:
    print(f"\nTop Prediction: {top_pred.species_name} ({top_pred.confidence:.4f})")

# Print the top 3 most likely species
print("\nTop 3 Predictions:")
for prediction in result.predictions[:3]:
    print(f"- {prediction.species_name}: {prediction.confidence:.4f}")
```

### 4.3. Interpreting and Visualizing Classification Results

A bar chart is a great way to understand the model's confidence across all potential species. Let's visualize the top 5 predictions using the structured prediction format.

```python
# Extract the names and probabilities of the top 5 predictions for our chart
top_5_predictions = result.predictions[:5]
species_names = [pred.species_name for pred in top_5_predictions]
probabilities = [pred.confidence for pred in top_5_predictions]

# Create a horizontal bar plot
plt.figure(figsize=(10, 6))
plt.barh(species_names, probabilities, color="skyblue")
plt.xlabel("Probability")
plt.title("Species Classification Probabilities (Top 5)")
plt.gca().invert_yaxis()  # Invert axis to show the most likely result at the top

# Add the probability values as text on the bars for clarity
for index, value in enumerate(probabilities):
    plt.text(value, index, f" {value:.2%}", va='center')

plt.tight_layout()
plt.show()
```

## 5. Mosquito Segmentation

Segmentation goes one step further than detection. Instead of just a box, it provides a precise, pixel-level mask outlining the exact shape of the mosquito.

### 5.1. Initializing the Segmenter

Again, we initialize our `MosquitoSegmenter` from the `settings` object.

```python
# Initialize the segmenter
print("\nInitializing MosquitoSegmenter...")
segmenter = MosquitoSegmenter(settings=settings, load_model=True)
print("Segmenter model loaded and ready.")
```

### 5.2. Performing Segmentation

The `predict` method returns a structured `SegmentationPrediction` object containing the binary mask and additional metadata. The mask is a 2D numpy array where pixels belonging to the mosquito are marked as `True` (or `255`), while background pixels are `False` (or `0`).

There are two ways to perform segmentation:

#### **Method 1: Basic Segmentation**

You can run the segmenter on the whole image. It will try to find and segment the most prominent object.

```python
print("\n--- Performing basic segmentation on the full image ---")
basic_result = segmenter.predict(image_rgb)
basic_mask = basic_result.mask
print("Basic segmentation complete.")
```

#### **Method 2: Detection-Guided Segmentation (Recommended)**

For the best results, you can pass the detection result obtained from the `MosquitoDetector`. This tells the segmenter exactly where to look, resulting in a more accurate and cleaner mask.

```python
# We'll use the detection results from the detector earlier
# Convert detections to the format expected by the segmenter
detection_boxes = [detection.box.to_numpy() for detection in result.detections]

print("\n--- Performing segmentation using detection boxes as a guide ---")
guided_result = segmenter.predict(image_rgb, detection_boxes=detection_boxes)
guided_mask = guided_result.mask
print("Guided segmentation complete.")
```

### 5.3. Visualizing Segmentation Results

The `.visualize()` method is perfect for seeing the final mask overlaid on the original image.

```python
# Visualize the more accurate, guided segmentation result
segmented_image = segmenter.visualize(image_rgb, guided_result)

# Display the original image, the mask itself, and the final overlay
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(guided_mask, cmap="gray")
plt.axis("off")
plt.title("Segmentation Mask (Guided)")

plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.axis("off")
plt.title("Segmented Overlay")

plt.tight_layout()
plt.show()
```

## 6. Advanced Features and Production Deployment

### 6.1. High-Performance Serve API

For production applications and high-throughput scenarios, CulicidaeLab provides a dedicated `serve` API that offers significant performance improvements through automatic ONNX backend selection and intelligent caching.

#### Basic Serve API Usage

```python
from culicidaelab.serve import serve, clear_serve_cache

# The serve API accepts the same image formats as regular predictors:
# - File paths (str or Path): "image.jpg"
# - PIL Images: Image.open("image.jpg")
# - NumPy arrays: np.array(image)
# - Image bytes: image_bytes

# Fast classification - automatically uses ONNX backend
result = serve("mosquito.jpg", predictor_type="classifier")
species = result.top_prediction().species_name
confidence = result.top_prediction().confidence

print(f"Species: {species} (Confidence: {confidence:.2%})")

# Fast detection with confidence threshold
result = serve("image.jpg", predictor_type="detector", confidence_threshold=0.7)
print(f"Found {len(result.detections)} mosquitoes")

# Fast segmentation
result = serve("image.jpg", predictor_type="segmenter")
mask = result.mask
print(f"Segmentation mask shape: {mask.shape}")

# Clean up resources when done
clear_serve_cache()
```

#### Performance Comparison

The serve API provides substantial performance improvements, especially for repeated predictions:

```python
import time
from culicidaelab.serve import serve, clear_serve_cache

# Traditional approach
classifier = MosquitoClassifier(settings=settings, load_model=True)
start = time.time()
result1 = classifier.predict("image.jpg")
traditional_time = time.time() - start

# Serve API - first call (loads model)
start = time.time()
result2 = serve("image.jpg", predictor_type="classifier")
serve_first_time = time.time() - start

# Serve API - subsequent calls (uses cache)
start = time.time()
result3 = serve("image.jpg", predictor_type="classifier")
serve_cached_time = time.time() - start

print(f"Traditional: {traditional_time:.3f}s")
print(f"Serve (first): {serve_first_time:.3f}s")
print(f"Serve (cached): {serve_cached_time:.3f}s")  # Typically 10-100x faster

clear_serve_cache()
```

#### Web API Integration

The serve API is perfect for web services and REST APIs:

```python
from fastapi import FastAPI, UploadFile
from culicidaelab.serve import serve
import json

app = FastAPI()

@app.post("/predict/{predictor_type}")
async def predict(predictor_type: str, file: UploadFile):
    # Read uploaded image
    image_bytes = await file.read()
    
    # Run prediction using serve API
    result = serve(image_bytes, predictor_type=predictor_type)
    
    # Return structured JSON response
    return json.loads(result.model_dump_json())

@app.on_event("shutdown")
async def shutdown():
    clear_serve_cache()
```

### 6.2. Utility Functions for Discovery

CulicidaeLab provides utility functions to programmatically discover available models and datasets:

```python
from culicidaelab import list_models, list_datasets

# Discover available models
available_models = list_models()
print("Available models:")
for model in available_models:
    print(f"  - {model}")

# Discover available datasets
available_datasets = list_datasets()
print("\nAvailable datasets:")
for dataset in available_datasets:
    print(f"  - {dataset}")
```

### 6.3. Memory Management and Context Managers

For memory-efficient processing, use context managers to automatically load and unload models:

```python
# Temporary model loading - automatically unloads after use
with classifier.model_context():
    result = classifier.predict(image_rgb)
    # Process result here
# Model automatically unloaded after context

# Batch processing with automatic cleanup
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = []

with classifier.model_context():
    for image_path in images:
        result = classifier.predict(image_path)
        results.append(result)
# Model unloaded after processing all images
```

### 6.4. Batch Processing and Evaluation

For advanced users who need to process large datasets or assess model performance, each predictor supports enhanced batch operations:

#### Enhanced Batch Processing

```python
# Batch processing with progress tracking
images = [image1, image2, image3, ...]  # List of images
results = classifier.predict_batch(
    input_data_batch=images,
    show_progress=True  # Shows progress bar
)

# Process results
for i, result in enumerate(results):
    top_pred = result.top_prediction()
    print(f"Image {i+1}: {top_pred.species_name} ({top_pred.confidence:.2%})")
```

#### Model Evaluation

```python
# Evaluate model performance against ground truth
ground_truths = ["species1", "species2", "species3", ...]  # True labels

evaluation_report = classifier.evaluate_batch(
    input_data_batch=images,
    ground_truth_batch=ground_truths,
    show_progress=True
)

# Display evaluation metrics
print("\nEvaluation Results:")
for metric, value in evaluation_report.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")

# Visualize confusion matrix and metrics
classifier.visualize_report(evaluation_report)
```

### 6.5. Production Deployment Best Practices

#### Resource Management
- Use `clear_serve_cache()` to free GPU memory when switching between different predictor types
- Implement proper error handling for production environments
- Monitor memory usage in long-running applications

#### Performance Optimization
- Use the serve API for production deployments
- Batch process multiple images when possible
- Consider using context managers for temporary model loading

#### Scalability
- The serve API's caching mechanism makes it ideal for web services
- Structured outputs are JSON-serializable for easy API responses
- ONNX backends provide consistent performance across different hardware

For more detailed information about advanced features, deployment strategies, and API reference, please refer to the complete documentation and explore the code examples provided in the repository.
