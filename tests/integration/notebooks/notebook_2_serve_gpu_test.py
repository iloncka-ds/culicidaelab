# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Test for `culicidaelab` Serve GPU Installation
#
# **Variant:** `pip install "culicidaelab[serve-gpu]"`
#
# **Purpose:** This notebook tests lightweight, GPU-accelerated inference
# and core library functionalities.

# %%
# Step 1: All imports at the top of the file
import matplotlib.pyplot as plt
import torch

from culicidaelab import (
    DatasetsManager,
    get_settings,
    serve,
    clear_serve_cache,
)

# %%
# Step 2: Install the library
print("--- Installing culicidaelab[serve-gpu] ---")
# Uncomment the following line if you have not yet installed the library
# !pip install -q "culicidaelab[serve-gpu]"
print("--- Installation complete ---")

# %%
# Step 3: Verify GPU
if torch.cuda.is_available():
    print(f"✅ GPU is available. Device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: GPU not available. Inference will fall back to CPU.")

# %%
# Step 4: Test Settings
print("\n--- Testing Settings ---")
settings = get_settings()
print(f"Active Config Directory: {settings.config_dir}")
print("Settings test successful.")

# %%
# Step 5: Test Datasets
print("\n--- Testing Datasets ---")
manager = DatasetsManager(settings)
classification_dataset = manager.load_dataset("classification", split="test[:5]")
print(f"Loaded {len(classification_dataset)} classification samples.")
sample = classification_dataset[0]
print(f"Sample label: {sample['label']}")
print("Datasets test successful.")

# %%
# Step 6: Test Serve Module
print("\n--- Testing Serve Module (GPU Inference) ---")
image = sample["image"]
ground_truth_label = sample["label"]

# Display the input image
plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.title(f"Input Image\n(Ground Truth: {ground_truth_label})")
plt.axis("off")
plt.show()

# Run classification using the serve function
# It should automatically use the GPU provider if onnxruntime-gpu is installed
print("\n--- Running classification via serve() ---")
classification_result = serve(image, predictor_type="classifier")

print("\n--- Top 3 Classification Predictions ---")
for p in classification_result.predictions[:3]:
    print(f"{p.species_name}: {p.confidence:.2%}")

# Clean up
clear_serve_cache()
print("Serve module test successful.")

# %%
print("\n✅ Serve GPU installation test completed successfully.")
