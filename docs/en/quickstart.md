# Quick Start Guide

This guide is designed to get you from zero to your first mosquito classification in under five minutes. We will walk through a single, complete code example that you can copy, paste, and run to see `CulicidaeLab` in action.

The example will show you how to:

1.  Initialize the library using the central `settings` object.
2.  Load an image of a mosquito.
3.  Run the `MosquitoClassifier` to predict its species.
4.  Visualize the result.

### Prerequisites

Before you run the code, you need to have:

1.  **`CulicidaeLab` installed.** If you haven't installed it yet, please follow the **[Installation Guide](./installation.md)**.
2.  **A test image.** Create a folder named `test_imgs` in your project's root directory. Place an image of a mosquito inside it and name it `mosquito.jpg`.

Your project structure should look like this:
```
your_project/
├── test_imgs/
│   └── mosquito.jpg
└── your_script.py
```

---

### Complete Example: From Image to Classification

Copy the code block below into your Python script or a Jupyter Notebook and run it. The library will automatically download and cache the required model on the first run.

```python
# 1. Imports: Get all the necessary tools
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import the main entry point and the classifier from CulicidaeLab
from culicidaelab import get_settings
from culicidaelab.predictors import MosquitoClassifier

# --- Main script ---

# 2. Initialization: Set up the library and the classifier
print("Initializing CulicidaeLab...")
# Get the central settings object, which manages all configurations.
settings = get_settings()
# Ask the settings object to create a classifier. This is the recommended way.
classifier = MosquitoClassifier(settings=settings)
print("Classifier ready.")


# 3. Load Image: Prepare your data for prediction
print("Loading image...")
image_path = Path("test_imgs") / "mosquito.jpg"
try:
    # Use OpenCV to read the image file
    image = cv2.imread(str(image_path))
    # The library expects images in RGB format, so we convert from OpenCV's default BGR
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Image loaded successfully.")
except Exception as e:
    print(f"Error: Could not load image from {image_path}.")
    print("Please make sure the file exists and the path is correct.")
    exit()


# 4. Run Prediction: Get the classification result
print("Running prediction...")
# The `model_context()` manager handles loading/unloading the model from memory efficiently.
with classifier.model_context():
    predictions = classifier.predict(image_rgb)
print("Prediction complete.")

# The `predictions` object is a list of (species_name, confidence_score) tuples,
# sorted from most to least likely.
top_prediction = predictions[0]
print(f"\n---> Top Result: {top_prediction[0]} (Confidence: {top_prediction[1]:.2%})")


# 5. Visualize the Result: See the prediction on the image
print("\nVisualizing result...")
# The `.visualize()` method draws the top predictions directly onto the image.
with classifier.model_context():
    annotated_image = classifier.visualize(image_rgb, predictions)

# Display the original and annotated images side-by-side for comparison
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(annotated_image)
plt.title("Classification Result")
plt.axis("off")

plt.tight_layout()
plt.show()
print("Done!")
```
