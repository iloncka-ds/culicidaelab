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
# # Test for `culicidaelab` Full CPU Installation
#
# **Variant:** `pip install "culicidaelab[full]" --extra-index-url https://download.pytorch.org/whl/cpu`
#
# **Purpose:** To verify the complete development and research environment on a
# CPU, covering all library functionalities from settings to model evaluation.

# %%
# Step 1: All imports at the top of the file

from culicidaelab import (
    DatasetsManager,
    MosquitoClassifier,
    MosquitoDetector,
    MosquitoSegmenter,
    clear_serve_cache,
    get_settings,
    serve,
)

# %%
# Step 2: Install the library
print("--- Installing culicidaelab[full] for CPU ---")
# Uncomment the following line if you have not yet installed the library
# !pip install -q "culicidaelab[full]" --extra-index-url https://download.pytorch.org/whl/cpu
print("--- Installation complete ---")

# %%
# Step 3: Initialize Settings for CPU
print("\n--- Initializing Core Components for CPU ---")
settings = get_settings()
settings.set_config("predictors.classifier.device", "cpu")
settings.set_config("predictors.detector.device", "cpu")
settings.set_config("predictors.segmenter.device", "cpu")
print("✅ Settings initialized. Device for all predictors set to 'cpu'")

# %%
# Step 4: Test Datasets
print("\n--- Testing Datasets ---")
manager = DatasetsManager(settings)
class_data = manager.load_dataset("classification", split="test[:5]")
detect_data = manager.load_dataset("detection", split="train[:5]")
print("Datasets loaded successfully.")

# %%
# Step 5: Test Classification
print("\n--- Testing Classification ---")
classifier = MosquitoClassifier(settings, load_model=True)
class_sample = class_data[0]
result_class = classifier.predict(class_sample["image"])
print(f"Top classification prediction: {result_class.predictions[0].species_name}")

# %%
# Step 6: Test Detection
print("\n--- Testing Detection ---")
detector = MosquitoDetector(settings, load_model=True)
detect_sample = detect_data[0]
result_detect = detector.predict(detect_sample["image"])
print(f"Detections found: {len(result_detect.detections)}")

# %%
# Step 7: Test Segmentation
print("\n--- Testing Segmentation ---")
segmenter = MosquitoSegmenter(settings, load_model=True)
# Use the detection result as a prompt for segmentation
detection_boxes = [d.box.to_numpy() for d in result_detect.detections]
if detection_boxes:
    result_seg = segmenter.predict(detect_sample["image"], detection_boxes=detection_boxes)
    print(f"Segmentation mask created with {result_seg.pixel_count} pixels.")
else:
    print("Skipping segmentation as no objects were detected.")

# %%
# Step 8: Test Serve Module
print("\n--- Testing Serve Module ---")
serve_result = serve(class_sample["image"], predictor_type="classifier")
print(f"Top serve prediction: {serve_result.predictions[0].species_name}")
clear_serve_cache()

# %%
print("\n✅ Full CPU installation test completed successfully.")
