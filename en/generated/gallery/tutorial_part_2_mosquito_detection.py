# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: culicidaelab (3.11.6)
#     language: python
#     name: python3
# ---

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

# %% [markdown]
# Install the `culicidaelab` library if not already installed
# ```bash
# !pip install -q culicidaelab[full]
# ```
# or, if you have access to GPU
# ```bash
# !pip install -q culicidaelab[full-gpu]
# `

# %% [markdown]
# ## 1. Initialization
#
# First, we'll get the global `settings` instance and use it to initialize our `MosquitoDetector`.
# By setting `load_model=True`, we tell the detector to load the model weights into memory immediately.
# If the model file doesn't exist locally, it will be downloaded automatically.

# %%

from PIL import Image, ImageDraw, ImageFont
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
result = detector.predict(detect_image)

# The `visualize` method draws the bounding boxes onto the image for easy inspection.
annotated_image = detector.visualize(detect_image, result)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Detected Mosquitoes")
plt.show()

# Print the numerical detection results
print("\nDetection Results:")
if result:
    for i, det in enumerate(result.detections):
        print(
            f"  - Mosquito {i+1}: \
            Confidence = {det.confidence:.2f}, \
            Box = (x1={det.box.x1:.1f}, y1={det.box.y1:.1f}, x2={det.box.x2:.1f}, y2={det.box.y2:.1f})",
        )
else:
    print("  No mosquitoes detected.")

# %% [markdown]
# ## 3. Evaluating a Prediction with Ground Truth
#
# The `evaluate` method allows you to compare a prediction against a ground truth.
# This is useful for measuring the model's accuracy. The method returns several metrics,
# which are a standard for object detection.
# Now let's evaluate the prediction against the actual ground truth from the dataset.

# %%
# Extract ground truth boxes from the dataset sample
ground_truth_boxes = []
for bbox in objects["bboxes"]:
    x_min, y_min, x_max, y_max = bbox
    ground_truth_boxes.append((x_min, y_min, x_max, y_max))

# Evaluate using the ground truth from the dataset
print("--- Evaluating with dataset ground truth ---")
evaluation = detector.evaluate(ground_truth=ground_truth_boxes, prediction=result)
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
    print(f"  - Image {i+1}: Found {len(dets.detections)} detection(s).")

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
def visualize_comparison(image_rgb, ground_truth_boxes, detections):
    """
    Visualize ground truth and detection bounding boxes on an image using Pillow.

    Args:
        image_rgb: RGB image as numpy array or PIL Image
        ground_truth_boxes: List of ground truth bounding boxes [x_min, y_min, x_max, y_max]
        detections: List of detections results

    Returns:
        PIL Image with bounding boxes drawn
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image_rgb, np.ndarray):
        if image_rgb.dtype == np.float32 or image_rgb.dtype == np.float64:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        image = Image.fromarray(image_rgb)
    else:
        image = image_rgb.copy()

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Try to load a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Draw ground truth boxes in green
    for bbox in ground_truth_boxes:
        x_min, y_min, x_max, y_max = (int(v) for v in bbox)

        # Draw rectangle
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline="green",
            width=2,
        )

        # Draw label
        text = "GT"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position text above the box
        text_x = x_min
        text_y = max(0, y_min - text_height - 2)

        # Draw text background
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width, text_y + text_height)],
            fill="green",
        )

        # Draw text
        draw.text((text_x, text_y), text, fill="white", font=font)

    # Draw detection boxes in blue with confidence
    for det in detections.detections:
        x_min, y_min, x_max, y_max = int(det.box.x1), int(det.box.y1), int(det.box.x2), int(det.box.y2)

        # Draw rectangle
        draw.rectangle(
            [(x_min, y_min), (x_max, y_max)],
            outline="red",
            width=2,
        )

        # Draw confidence label
        text = f"{det.confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position text above the box
        text_x = x_min
        text_y = max(0, y_min - text_height - 2)

        # Draw text background
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width, text_y + text_height)],
            fill="red",
        )

        # Draw text
        draw.text((text_x, text_y), text, fill="white", font=font)

    return image


# Create comparison visualization
comparison_image = visualize_comparison(np.array(detect_image), ground_truth_boxes, result)

# Display the comparison
plt.figure(figsize=(12, 8))
plt.imshow(comparison_image)
plt.axis("off")
plt.title("Ground Truth vs Predictions\nGreen: Ground Truth\nRed: Predictions with Confidence")
plt.show()
