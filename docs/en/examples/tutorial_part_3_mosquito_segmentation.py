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
# Mosquito Segmentation Tutorial

This tutorial demonstrates how to use the `culicidaelab` library
to perform mosquito segmentation on images. We'll cover:

1. Setting up the segmentation model
2. Loading segmentation data from the dataset
3. Running segmentation
4. Visualizing results
5. Evaluating performance with ground truth masks

"""

# %%
# Install the `culicidaelab` library if not already installed
# # !pip install -q culicidaelab

# %%
import matplotlib.pyplot as plt
import numpy as np

from culicidaelab import MosquitoSegmenter, MosquitoDetector
from culicidaelab import DatasetsManager, get_settings

# %% [markdown]
# ## 1. Initialize Settings and Load Dataset
#
# First, we'll initialize our settings, create MosquitoSegmenter and load the segmentation dataset:

# %%
# Get settings instance and initialize dataset manager
settings = get_settings()
manager = DatasetsManager(settings)

# Load segmentation dataset
seg_data = manager.load_dataset("segmentation", split="train[:20]")

# Initialize segmenter and detector
segmenter = MosquitoSegmenter(settings=settings, load_model=True)
detector = MosquitoDetector(settings=settings, load_model=True)

# %% [markdown]
# ## 2. Inspect a Segmentation Sample
#
# Let's examine a sample from the segmentation dataset to understand its structure:

# %%
# Inspect a segmentation sample
seg_sample = seg_data[0]
seg_image = seg_sample["image"]
seg_mask = np.array(seg_sample["label"])  # Convert mask to numpy array

print(f"Image size: {seg_image.size}")
print(f"Segmentation mask shape: {seg_mask.shape}")
print(f"Unique values in mask: {np.unique(seg_mask)}")  # 0 is background, 1 and above is mosquito

# Create a colored overlay for the mask
# Where the mask is 1 and above (mosquito), we make it red
overlay = np.zeros((*seg_mask.shape, 4), dtype=np.uint8)
overlay[seg_mask >= 1] = [255, 0, 0, 128]  # Red color with 50% opacity

# %% [markdown]
# ## 3. Run Segmentation on Dataset Image
#
# Now we can run the segmentation model on our dataset image:

# %%
# Run detection to get bounding boxes
result = detector.predict(seg_image)
bboxes = [detection.box.to_numpy() for detection in result.detections]
# Run segmentation with detection boxes
predicted_mask = segmenter.predict(seg_image, detection_boxes=np.array(bboxes))

# Create visualizations
annotated_image = detector.visualize(seg_image, result)
segmented_image = segmenter.visualize(annotated_image, predicted_mask)

# %% [markdown]
# ## 4. Visualize Results with Ground Truth Comparison
#
# Let's visualize the segmentation results alongside the ground truth mask:

# %%
plt.figure(figsize=(20, 10))

# Original image
plt.subplot(2, 4, 1)
plt.imshow(seg_image)
plt.axis("off")
plt.title("Original Image")

# Ground truth mask
plt.subplot(2, 4, 2)
plt.imshow(seg_mask, cmap="gray")
plt.axis("off")
plt.title("Ground Truth Mask")

# Ground truth overlay
plt.subplot(2, 4, 3)
plt.imshow(seg_image)
plt.imshow(overlay, alpha=0.5)
plt.axis("off")
plt.title("Ground Truth Overlay")

# Detections
plt.subplot(2, 4, 4)
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Detected Mosquitoes")

# Predicted mask
plt.subplot(2, 4, 5)
plt.imshow(predicted_mask.mask, cmap="gray")
plt.axis("off")
plt.title("Predicted Mask")

# Predicted overlay
predicted_overlay = np.zeros((*predicted_mask.mask.shape, 4), dtype=np.uint8)
predicted_overlay[predicted_mask.mask >= 0.5] = [0, 255, 0, 128]  # Green for predictions
plt.subplot(2, 4, 6)
plt.imshow(seg_image)
plt.imshow(predicted_overlay, alpha=0.5)
plt.axis("off")
plt.title("Predicted Overlay")

# Combined overlay (ground truth + predictions)
combined_overlay = np.zeros((*predicted_mask.mask.shape, 4), dtype=np.uint8)
combined_overlay[seg_mask >= 1] = [255, 0, 0, 128]  # Red for ground truth
combined_overlay[predicted_mask.mask >= 0.5] = [0, 255, 0, 128]  # Green for predictions
plt.subplot(2, 4, 7)
plt.imshow(seg_image)
plt.imshow(combined_overlay, alpha=0.5)
plt.axis("off")
plt.title("Combined Overlay\n(Red: GT, Green: Pred)")

# Final segmented image
plt.subplot(2, 4, 8)
plt.imshow(segmented_image)
plt.axis("off")
plt.title("Final Segmented Image")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Evaluate Segmentation Performance
#
# Let's evaluate the segmentation results using the ground truth mask:

# %%
metrics = segmenter.evaluate(
    prediction=predicted_mask,
    ground_truth=seg_mask,
)
print("Segmentation Evaluation Metrics:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
