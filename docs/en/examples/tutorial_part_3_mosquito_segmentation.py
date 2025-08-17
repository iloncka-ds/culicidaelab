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
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

# %% [markdown]
# ## 3. Run Segmentation
#
# Now we can run the segmentation model on our image:

# %%
# Initialize detector
detector = MosquitoDetector(settings=settings, load_model=True)
# Load test image
image_path = str(Path("test_imgs") / "640px-Aedes_aegypti.jpg")


# Run detection
detections = detector.predict(image_path)

# Run segmentation with detection boxes
mask = segmenter.predict(image_path, detection_boxes=np.array(detections))
# Draw detections
annotated_image = detector.visualize(image_path, detections)
segmented_image = segmenter.visualize(annotated_image, mask)

# %% [markdown]
# ## 4. Visualize Results
#
# Finally, let's visualize the segmentation results overlaid on the original image:

# %%
image = Image.open(image_path)
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
# ## 5. Evaluate results

# %%
metrics = segmenter.evaluate(mask, input_data=image_path, detection_boxes=detections)
print(metrics)
