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
#  Using the Serve Module for Production Inference

This tutorial demonstrates how to use the `serve` function from the CulicidaeLab
library for high-performance, production-ready inference. The `serve` function
is designed to be a lightweight, fast, and safe way to run predictions.

This guide will cover:

- **Speed and Safety**: How the `serve` function uses the ONNX backend for fast inference.
- **Single Image Prediction**: How to use `serve` for classification.
- **Caching**: Understand the in-memory caching for predictor instances.
- **Clearing the Cache**: How to clear the cache when needed.
"""

# %% [markdown]
# Install the `culicidaelab` library if not already installed
# ```bash
# !pip install -q culicidaelab[full]
# ```
# ## 1. Initialization and Setup
#
# We will initialize the `DatasetsManager` to get some sample data.
# The `serve` function doesn't require manual initialization of predictors.

# %%
# Import necessary libraries
import matplotlib.pyplot as plt

# Import the required classes from the CulicidaeLab library
from culicidaelab import (
    DatasetsManager,
    get_settings,
    serve,
    clear_serve_cache,
)

# Get the default library settings instance
settings = get_settings()

# Initialize the services needed to manage and download data
manager = DatasetsManager(settings)

# %% [markdown]
# ## 2. Loading the Test Dataset
#
# We will use a built-in test dataset to get an image for our predictions.

# %%
print("--- Loading the 'classification' dataset's 'test' split ---")
classification_test_data = manager.load_dataset("classification", split="test")
print("Test dataset loaded successfully!")
print(f"Number of samples in the test dataset: {len(classification_test_data)}")

# Let's select one sample to work with.
classification_test_data = classification_test_data.shuffle(seed=42)
sample = classification_test_data[0]
image = sample["image"]
ground_truth_label = sample["label"]

print(f"\nSelected sample's ground truth label: '{ground_truth_label}'")

# Display the input image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Input Image\n(Ground Truth: {ground_truth_label})")
plt.axis("off")
plt.show()


# %% [markdown]
# ## 3. Using `serve` for Classification
#
# The `serve` function automatically initializes the predictor with the ONNX backend
# on the first call and caches it for subsequent requests.

# %%
# Run classification using the serve function
print("--- Running classification for the first time (will initialize predictor) ---")
classification_result = serve(image)

# Print the top 5 predictions
print("\n--- Top 5 Classification Predictions ---")
for p in classification_result.predictions[:5]:
    print(f"{p.species_name}: {p.confidence:.2%}")

# %% [markdown]
# ## 4. Caching in Action
#
# If you run the same request again, you'll notice it's much faster because
# the predictor is already in memory.

# %%
# Run classification again to see the caching effect
print("\n--- Running classification for the second time (should be faster) ---")
classification_result_cached = serve(image, predictor_type="classifier")

# Print the top 5 predictions again
print("\n--- Top 5 Classification Predictions (from cache) ---")
for p in classification_result_cached.predictions[:5]:
    print(f"{p.species_name}: {p.confidence:.2%}")


# %% [markdown]
# ## 5. Clearing the Cache
#
# If you need to free up memory or reload the predictors, you can use the
# `clear_serve_cache` function.

# %%
# Clear the cache
print("\n--- Clearing the predictor cache ---")
clear_serve_cache()

# Run classification again, it will re-initialize the predictor
print("\n--- Running classification again after clearing cache (will re-initialize) ---")
classification_result_after_clear = serve(image)

print("\n--- Top 5 Classification Predictions (after cache clear) ---")
for p in classification_result_after_clear.predictions[:5]:
    print(f"{p.species_name}: {p.confidence:.2%}")
