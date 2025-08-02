# %%
"""
# Mosquito Segmentation Tutorial

This tutorial demonstrates how to use the CulicidaeLab library
to perform mosquito segmentation on images. We'll cover:

1. Setting up the segmentation model
2. Loading and preprocessing images
3. Running segmentation
4. Visualizing results

"""

# %%
# Install the `culicidaelab` library if not already installed
# # !pip install -q culicidaelab

# %%
# First, let's import the necessary libraries:
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from culicidaelab import MosquitoSegmenter, MosquitoDetector

from culicidaelab import ModelWeightsManager
from culicidaelab import get_settings

# %% [markdown]
# ## 1. Initialize Settings and Segmenter
#
# First, we'll initialize our settings and create a MosquitoSegmenter instance:

# %%
# Get settings instance
settings = get_settings()
settings.list_model_types()

# %%
model_config = settings.get_config("predictors.segmenter")
model_path = settings.get_model_weights_path("segmenter")
weights_manager = ModelWeightsManager(settings=settings)
# Initialize segmenter
segmenter = MosquitoSegmenter(settings=settings, load_model=True)

# %% [markdown]
# ## 2. Load and Preprocess Image
#
# Now let's load a test image:

# %%
# Load test image
image_path = str(Path("test_imgs") / "640px-Aedes_aegypti.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %% [markdown]
# ## 3. Run Segmentation
#
# Now we can run the segmentation model on our image:

# %%
mask = segmenter.predict(image)

# %% [markdown]
# ## 4. Visualize Results
#
# Finally, let's visualize the segmentation results overlaid on the original image:

# %%
# Visualize segmentation results
segmented_image = segmenter.visualize(image, mask)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.title("Segmentation Mask")

plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.axis("off")
plt.title("Segmented Image")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Segmentation Using Detection Results
#

# %% [markdown]
# The segmenter can also use detection results to improve segmentation accuracy.
# Here's how to combine detection and segmentation:

# %%
# Initialize detector
detector = MosquitoDetector(settings=settings, load_model=True)
# Load test image
image_path = str(Path("test_imgs") / "640px-Aedes_aegypti.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
detections = detector.predict(image)

# Run segmentation with detection boxes
mask_with_boxes = segmenter.predict(image, detection_boxes=detections)
# Draw detections
annotated_image = detector.visualize(image, detections)

# Print detection results
print("\nDetection Results:")
for i, (x, y, w, h, conf) in enumerate(detections):
    print(
        f"Mosquito {i+1}: Confidence = {conf:.2f}, Box = (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})",
    )
# Visualize results
segmented_image_with_boxes = segmenter.visualize(annotated_image, mask_with_boxes)

# plt.figure(figsize=(10, 10))
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(mask_with_boxes, cmap="gray")
plt.axis("off")
plt.title("Segmentation Mask")


plt.subplot(1, 2, 2)
plt.imshow(segmented_image_with_boxes)
plt.axis("off")
plt.title("Segmentation with Detection Boxes")

plt.tight_layout()
plt.show()

# %%
metrics = segmenter.evaluate(mask_with_boxes, input_data=image, detection_boxes=detections)
print(metrics)

# %%
metrics_default = segmenter.evaluate(
    mask_with_boxes,
    mask,
)
print(metrics_default)
