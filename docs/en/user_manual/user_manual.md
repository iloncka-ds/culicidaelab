
# CulicidaeLab User Guide

Welcome to the `CulicidaeLab` User Guide! This guide will walk you through the essential features of the library, showing you how to perform mosquito detection, classification, and segmentation. We will cover everything from initial setup to running predictions and visualizing the results.

## 1. Before You Begin: Core Concepts

### The `settings` Object

The single most important concept in `CulicidaeLab` is the **`settings` object**. It is your centralized entry point for everything. Instead of importing and initializing each component manually with complex parameters, you simply:

1. Get the `settings` object.
2. Ask it to create the component you need (`Detector`, `Classifier`, etc.).

The `settings` object automatically handles loading configurations, managing file paths, and downloading models.

### The Predictor Workflow

All three main components (`MosquitoDetector`, `MosquitoClassifier`, `MosquitoSegmenter`) are **Predictors**. They share a consistent and predictable workflow:

1. **Initialize**: Create an instance from the `settings` object.
2. **Load Model**: The model weights are lazy-loaded, meaning they are only downloaded and loaded into memory on the first prediction or when you explicitly tell them to. We will use `load_model=True` for clarity.
3. **Predict**: Use the `.predict()` method on an image.
4. **Visualize**: Use the `.visualize()` method to see the results.

Let's get started!

## 2. Initialization and Setup

First, let's set up our environment and get the all-important `settings` object.

```python
# Import necessary libraries for image handling and plotting
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

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
image = cv2.imread(str(image_path))

# The library expects images in RGB format, but OpenCV loads them as BGR.
# We must convert the color space.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### 3.2. Performing Detection

The `predict` method takes an image as input and returns a list of all detected objects. Each object is a tuple containing the bounding box coordinates and a confidence score. The format is `(center_x, center_y, width, height, confidence)`.

```python
# Run the prediction on our RGB image
detections = detector.predict(image_rgb)

# Let's print the results in a human-readable format.
print("\nDetection Results:")
if detections:
    for i, (x, y, w, h, conf) in enumerate(detections):
        print(
            f"  - Mosquito {i+1}: Confidence = {conf:.2f}, Box = (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})"
        )
else:
    print("  No mosquitoes were detected in the image.")
```

### 3.3. Visualizing Detection Results

Reading coordinates is useful, but seeing the result is better. The `.visualize()` method draws the bounding boxes and confidence scores directly onto the image.

```python
# Pass the original image and the detections to the visualize method
annotated_image = detector.visualize(image_rgb, detections)

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

The classifier's `predict` method returns a list of all possible species, sorted by confidence score. Each item is a tuple of `(species_name, confidence_score)`.

```python
# Run the classification
predictions = classifier.predict(image_rgb)

# Print the top 3 most likely species
print("\nTop 3 Predictions:")
for species, confidence in predictions[:3]:
    print(f"- {species}: {confidence:.4f}")
```

### 4.3. Interpreting and Visualizing Classification Results

A bar chart is a great way to understand the model's confidence across all potential species. Let's visualize the top 5 predictions.

```python
# Extract the names and probabilities of the top 5 predictions for our chart
species_names = [p[0] for p in predictions[:5]]
probabilities = [p[1] for p in predictions[:5]]

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

The `predict` method returns a binary mask (a 2D numpy array). In this mask, pixels belonging to the mosquito are marked as `True` (or `255`), while background pixels are `False` (or `0`).

There are two ways to perform segmentation:

#### **Method 1: Basic Segmentation**

You can run the segmenter on the whole image. It will try to find and segment the most prominent object.

```python
print("\n--- Performing basic segmentation on the full image ---")
basic_mask = segmenter.predict(image_rgb)
print("Basic segmentation complete.")
```

#### **Method 2: Detection-Guided Segmentation (Recommended)**

For the best results, you can pass the bounding boxes obtained from the `MosquitoDetector`. This tells the segmenter exactly where to look, resulting in a more accurate and cleaner mask.

```python
# We'll use the 'detections' list we got from the detector earlier
print("\n--- Performing segmentation using detection boxes as a guide ---")
guided_mask = segmenter.predict(image_rgb, detection_boxes=detections)
print("Guided segmentation complete.")
```

### 5.3. Visualizing Segmentation Results

The `.visualize()` method is perfect for seeing the final mask overlaid on the original image.

```python
# Visualize the more accurate, guided mask
segmented_image = segmenter.visualize(image_rgb, guided_mask)

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

## 6. Next Steps: Batch Processing and Evaluation

For advanced users who need to process large datasets or assess model performance, each predictor also supports:

- **`.predict_batch()`**: Process a list of images at once for significantly better performance.
- **`.evaluate()` and `.evaluate_batch()`**: Compare model predictions against a dataset with known ground truth labels to calculate accuracy metrics like Average Precision (for detection) and IoU (for segmentation).

These methods are more complex and require data to be prepared in a specific format. For detailed information, please refer to the **API Reference** section of our documentation and explore the code examples provided in the repository.
