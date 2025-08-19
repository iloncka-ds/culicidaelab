# %%
"""
# Mosquito Detection Tutorial

This tutorial shows how to use the `MosquitoDetector` from the CulicidaeLab
library to perform object detection on images. We will cover:

- Loading the detector model
- Preparing an image from the dataset
- Running the model to get bounding boxes
- Visualizing the results
- Evaluating prediction accuracy
- Running predictions on a batch of images

"""

# %%
# Install the `culicidaelab` library if not already installed
# # !pip install -q culicidaelab

# %% [markdown]
# ## 1. Initialization
#
# First, we'll get the global `settings` instance and use it to initialize our `MosquitoDetector`.
# By setting `load_model=True`, we tell the detector to load the model weights into memory immediately.
# If the model file doesn't exist locally, it will be downloaded automatically.

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

from culicidaelab import get_settings
from culicidaelab import MosquitoDetector, DatasetsManager

# Get settings instance
settings = get_settings()

# Initialize the datasets manager
manager = DatasetsManager(settings)

# Load detection dataset
detect_data = manager.load_dataset("detection", split="train[:20]")

# Instantiate the detector and load the model
print("Initializing MosquitoDetector and loading model...")
detector = MosquitoDetector(settings=settings, load_model=True)
print("Model loaded successfully.")

# %% [markdown]
# ## 2. Detecting Mosquitoes in a Dataset Image
#
# Now let's use an image from the detection dataset and run the detector on it.

# %%
# Inspect a detection sample
detect_sample = detect_data[5]
detect_image = detect_sample["image"]

# Get ground truth objects
objects = detect_sample["objects"]
print(f"Found {len(objects['bboxes'])} object(s) in this image.")

# The `predict` method returns a list of detections.
# Each detection is a tuple: (x1, y1, x2, y2, confidence_score)
detections = detector.predict(detect_image)

# The `visualize` method draws the bounding boxes onto the image for easy inspection.
annotated_image = detector.visualize(detect_image, detections)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Detected Mosquitoes")
plt.show()

# Print the numerical detection results
print("\nDetection Results:")
if detections:
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        print(
            f"  - Mosquito {i+1}: \
            Confidence = {conf:.2f}, \
            Box = (x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f})",
        )
else:
    print("  No mosquitoes detected.")

# %% [markdown]
# ## 3. Evaluating a Prediction with Ground Truth
#
# The `evaluate` method allows you to compare a prediction against a ground truth.
# This is useful for measuring the model's accuracy. The method returns several metrics,
# including Average Precision (AP), which is a standard for object detection.
# Now let's evaluate the prediction against the actual ground truth from the dataset.

# %%
# Extract ground truth boxes from the dataset sample
ground_truth_boxes = []
for bbox in objects["bboxes"]:
    x_min, y_min, x_max, y_max = bbox
    ground_truth_boxes.append((x_min, y_min, x_max, y_max))

# Evaluate using the ground truth from the dataset
print("--- Evaluating with dataset ground truth ---")
evaluation = detector.evaluate(ground_truth=ground_truth_boxes, prediction=detections)
print(evaluation)

# %%
# You can let the method run prediction internally by passing the raw image
print("\n--- Evaluating directly from an image ---")
evaluation_from_raw = detector.evaluate(ground_truth=ground_truth_boxes, input_data=detect_image)
print(evaluation_from_raw)

# %% [markdown]
# ## 4. Running Batch Predictions on Dataset Images
#
# For efficiency, you can process multiple images at once using `predict_batch`.

# %%
# Extract images from the detection dataset
image_batch = [sample["image"] for sample in detect_data]

# Run batch prediction
detections_batch = detector.predict_batch(image_batch)
print("Batch prediction complete.")

for i, dets in enumerate(detections_batch):
    print(f"  - Image {i+1}: Found {len(dets)} detection(s).")

# %% [markdown]
# ## 5. Evaluating a Batch of Predictions with Dataset Ground Truth
#
# Similarly, `evaluate_batch` can be used to get aggregated metrics over the entire dataset.

# %%
# Extract ground truth from the detection dataset
ground_truth_batch = []
for sample in detect_data:
    boxes = []
    for bbox in sample["objects"]["bboxes"]:
        x_min, y_min, x_max, y_max = bbox
        boxes.append((x_min, y_min, x_max, y_max))
    ground_truth_batch.append(boxes)

# Call evaluate_batch with dataset ground truth
print("\n--- Evaluating the entire batch with dataset ground truth ---")
batch_evaluation = detector.evaluate_batch(
    ground_truth_batch=ground_truth_batch,
    predictions_batch=detections_batch,
    num_workers=2,  # Use multiple workers for faster processing
)

print("Aggregated batch evaluation metrics:")
print(batch_evaluation)

# %% [markdown]
# ## 6. Visualizing Ground Truth vs Predictions
#
# Let's create a comparison visualization showing both ground truth and predictions.


# %%
# Create a function to visualize both ground truth and predictions
def visualize_comparison(image_rgb, ground_truth_boxes, detection_boxes):
    # Draw ground truth boxes in green
    for bbox in ground_truth_boxes:
        x_min, y_min, x_max, y_max = (int(v) for v in bbox)
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            image_rgb,
            "GT",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    # Draw detection boxes in blue with confidence
    for x1, y1, x2, y2, conf in detection_boxes:
        x_min, y_min, x_max, y_max = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(
            image_rgb,
            f"{conf:.2f}",
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    return image_rgb


# Create comparison visualization
comparison_image = visualize_comparison(np.array(detect_image), ground_truth_boxes, detections)

# Display the comparison
plt.figure(figsize=(12, 8))
plt.imshow(comparison_image)
plt.axis("off")
plt.legend(["Green: Ground Truth", "Red: Predictions with Confidence"])
plt.title("Ground Truth vs Predictions\nGreen: Ground Truth\nRed: Predictions with Confidence")
plt.show()
