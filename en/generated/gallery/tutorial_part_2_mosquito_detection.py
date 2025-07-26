"""
# Mosquito Detection Tutorial

This tutorial shows how to use the `MosquitoDetector` from the CulicidaeLab
library to perform object detection on images. We will cover:

- Loading the detector model
- Preparing an image
- Running the model to get bounding boxes
- Visualizing the results
- Evaluating prediction accuracy
- Running predictions on a batch of images

"""
# %% [markdown]
# ## 1. Initialization
#
# First, we'll get the global `settings` instance and use it to initialize our `MosquitoDetector`.
# By setting `load_model=True`, we tell the detector to load the model weights into memory immediately.
# If the model file doesn't exist locally, it will be downloaded automatically.

# %%
import re
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from culicidaelab import get_settings
from culicidaelab import MosquitoDetector

# Get settings instance
settings = get_settings()

# Instantiate the detector and load the model
print("Initializing MosquitoDetector and loading model...")
detector = MosquitoDetector(settings=settings, load_model=True)
print("Model loaded successfully.")

# %% [markdown]
# ## 2. Detecting Mosquitoes in a Single Image
#
# Now let's load a test image and run the detector on it.

# %%
# Load a test image from the local 'test_imgs' directory
image_path = Path("test_imgs") / "640px-Aedes_aegypti.jpg"
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

# The `predict` method returns a list of detections.
# Each detection is a tuple: (center_x, center_y, width, height, confidence_score)
detections = detector.predict(image_rgb)

# The `visualize` method draws the bounding boxes onto the image for easy inspection.
annotated_image = detector.visualize(image_rgb, detections)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Detected Mosquitoes")
plt.show()

# Print the numerical detection results
print("\nDetection Results:")
if detections:
    for i, (x, y, w, h, conf) in enumerate(detections):
        print(f"  - Mosquito {i+1}: Confidence = {conf:.2f}, Box = (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})")
else:
    print("  No mosquitoes detected.")

# %% [markdown]
# ## 3. Evaluating a Prediction
#
# The `evaluate` method allows you to compare a prediction against a ground truth.
# This is useful for measuring the model's accuracy. The method returns several metrics,
# including Average Precision (AP), which is a standard for object detection.
#
# Here, we'll use the detection we just found as a mock ground truth to demonstrate the process.

# %%
# A ground truth is a list of boxes without the confidence score: [(x, y, w, h), ...]
if detections:
    test_ground_truth = [detections[0][:4]]  # Use the first detected box as our ground truth

    # You can evaluate using a pre-computed prediction
    print("--- Evaluating with a pre-computed prediction ---")
    evaluation = detector.evaluate(ground_truth=test_ground_truth, prediction=detections)
    print(evaluation)

    # Or you can let the method run prediction internally by passing the raw image
    print("\n--- Evaluating directly from an image ---")
    evaluation_from_raw = detector.evaluate(ground_truth=test_ground_truth, input_data=image_rgb)
    print(evaluation_from_raw)
else:
    print("Skipping evaluation as no detections were found.")


# %% [markdown]
# ## 4. Running Batch Predictions
#
# For efficiency, you can process multiple images at once using `predict_batch`.
# This is much faster than looping and calling `predict` on each image individually.

# %%
# Find all image files in the 'test_imgs' directory
image_dir = Path("test_imgs")
pattern = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)
image_paths = [path for path in image_dir.iterdir() if path.is_file() and pattern.search(str(path))]

# Load all images into a list (our "batch")
try:
    batch = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in image_paths]
    print(f"\n--- Processing a batch of {len(batch)} images ---")
except Exception as e:
    print(f"An error occurred while reading images: {e}")
    batch = []

# Run batch prediction
detections_batch = detector.predict_batch(batch)
print("Batch prediction complete.")
for i, dets in enumerate(detections_batch):
    print(f"  - Image {i+1} ({image_paths[i].name}): Found {len(dets)} detection(s).")


# %% [markdown]
# ## 5. Evaluating a Batch of Predictions
#
# Similarly, `evaluate_batch` can be used to get aggregated metrics over an entire set of images.

# %%
# Create a mock ground truth batch from our batch prediction results
batch_test_gt = [[(x, y, w, h) for (x, y, w, h, conf) in detections] for detections in detections_batch]

# Call evaluate_batch. We provide the predictions directly.
print("\n--- Evaluating the entire batch ---")
batch_evaluation = detector.evaluate_batch(
    ground_truth_batch=batch_test_gt,
    predictions_batch=detections_batch,
    num_workers=1,
)

print("Aggregated batch evaluation metrics:")
print(batch_evaluation)
