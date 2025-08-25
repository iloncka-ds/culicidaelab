# %%
"""
# Classifying Mosquito Species

This tutorial demonstrates how to use the `MosquitoClassifier` from the CulicidaeLab
library to identify mosquito species from images. We will walk through the entire
process, from loading the model to evaluating its performance on a batch of data.

This guide will cover:

- **Initialization**: How to load the settings and the pre-trained model.
- **Data Handling**: How to use the `DatasetsManager` to fetch sample data.
- **Single Image Prediction**: How to classify a single mosquito image.
- **Visualization**: How to interpret and visualize the model's predictions.
- **Batch Evaluation**: How to measure the model's accuracy on a set of test images.
- **Reporting**: How to generate and visualize a comprehensive performance report.
"""

# %%
# Install the `culicidaelab` library if not already installed
# !pip install -q culicidaelab

# %% [markdown]
# ## 1. Initialization and Setup
#
# Our first step is to set up the necessary components. We will initialize:
#
# - **`settings`**: An object that holds all library configuration, such as
#   model paths and hyperparameters.
# - **`DatasetsManager`**: A helper class to download and manage the sample
#   datasets used in this tutorial.
# - **`MosquitoClassifier`**: The main class for our classification task. We'll
#   pass `load_model=True` to ensure the pre-trained model weights are downloaded
#   and loaded into memory immediately.

# %%
# Import necessary libraries
import matplotlib.pyplot as plt

# Import the required classes from the CulicidaeLab library
from culicidaelab import (
    DatasetsManager,
    MosquitoClassifier,
    get_settings,
)

# Get the default library settings instance
settings = get_settings()

# Initialize the services needed to manage and download data

manager = DatasetsManager(settings)

# Instantiate the classifier and load the model.
# This might take a moment on the first run as it downloads the model weights.
print("Initializing MosquitoClassifier and loading model...")
classifier = MosquitoClassifier(settings, load_model=True)
print("Model loaded successfully.")

# %% [markdown]
# ### Inspecting Model Classes
#
# Before we start predicting, it's useful to know which species the model was
# trained to recognize. We can easily access this information from the settings
# object.

# %%
species_map = settings.species_config.species_map
print(f"--- The model can recognize {len(species_map)} classes ---")
# Print the first 5 for brevity
for idx, name in list(species_map.items())[:5]:
    print(f"  Class Index {idx}: {name}")
print("  ...")

# %% [markdown]
# ## 2. Loading the Test Dataset
#
# For this tutorial, we will use a built-in test dataset provided by the library.
# The `DatasetsManager` makes it simple to download and load this data. The dataset
# contains images and their corresponding correct labels, which we will use for
# prediction and later for evaluation.

# %%
print("\n--- Loading the 'classification' dataset's 'test' split ---")
classification_test_data = manager.load_dataset("classification", split="test")
print("Test dataset loaded successfully!")
print(f"Number of samples in the test dataset: {len(classification_test_data)}")

# Let's select one sample to work with.
# The sample is a dictionary containing the image and its ground truth label.
sample_index = 287
sample = classification_test_data[sample_index]
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
# ## 3. Classifying a Single Image
#
# Now we'll use the classifier to predict the species of the mosquito in our
# selected image. The `predict()` method takes an image (as a NumPy array, file
# path, or PIL Image) and returns a list of predictions, sorted from most to
# least confident.

# %%
# Run the classification on our sample image
predictions = classifier.predict(image)

# Print the top 5 predictions in a readable format
print("--- Top 5 Predictions ---")
for species, probability in predictions[:5]:
    print(f"{species}: {probability:.2%}")

# %% [markdown]
# ## 4. Visualizing and Interpreting the Results
#
# A raw list of predictions is useful, but visualizations make the results much
# easier to understand. We'll create two plots:
#
# 1.  **A Bar Plot**: This shows the model's confidence for every possible
#     species. It's great for seeing not just the top prediction, but also which
#     other species the model considered.
# 2.  **A Composite Image**: This uses the built-in `visualize()` method to create
#     a clean image that displays the top predictions alongside the input image.

# %%
# Create a bar plot to visualize the probabilities for all species
plt.figure(figsize=(10, 8))

# The predictions are already sorted, so we can plot them directly
species_names = [p[0] for p in predictions]
probabilities = [p[1] for p in predictions]

# We'll reverse the lists (`[::-1]`) so the highest probability is at the top
bars = plt.barh(species_names[::-1], probabilities[::-1])

# Highlight the bars that meet our confidence threshold
conf_threshold = settings.get_config("predictors.classifier.confidence")
for bar in bars:
    if bar.get_width() >= conf_threshold:
        bar.set_color("teal")
    else:
        bar.set_color("lightgray")

# Add a reference line for the confidence threshold
plt.axvline(
    x=conf_threshold,
    color="red",
    linestyle="--",
    label=f"Confidence Threshold ({conf_threshold:.0%})",
)
plt.xlabel("Assigned Probability")
plt.title("Species Classification Probabilities")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Now, let's use the built-in visualizer for a clean presentation
annotated_image = classifier.visualize(image, predictions)

# Display the final annotated image
plt.figure(figsize=(10, 6))
plt.imshow(annotated_image)
plt.title("Classification Result")
plt.axis("off")
plt.show()

# %% [markdown]
# ## 5. Evaluating Model Performance on a Batch
#
# While classifying a single image is useful, a more rigorous test involves
# evaluating the model's performance across an entire dataset. The
# `evaluate_batch()` method is designed for this. It processes a batch of
# images and their corresponding ground truth labels, then computes aggregate
# metrics.
#
# The result is a `report` dictionary containing key metrics like mean
# accuracy and a **confusion matrix**, which shows exactly where the model is
# succeeding or failing.

# %%
# Let's evaluate the first 30 images from the test set for this example
num_samples_to_evaluate = 30
batch_samples = classification_test_data.select(range(num_samples_to_evaluate))
batch_images = [sample["image"] for sample in batch_samples]
ground_truths = [sample["label"] for sample in batch_samples]

print(f"\n--- Evaluating a batch of {len(batch_images)} images ---")

# Run the batch evaluation.
# The method can take images and ground truths separately, or it can
# run predictions internally if you only provide the images.
report = classifier.evaluate_batch(
    input_data_batch=batch_images,
    ground_truth_batch=ground_truths,
    show_progress=True,
)

print("\n--- Evaluation Report Summary ---")
for key, value in report.items():
    if key != "confusion_matrix":
        # Check if value is a float before formatting
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


# %% [markdown]
# ## 6. Visualizing the Evaluation Report
#
# The generated `report` dictionary contains a wealth of information, but the
# confusion matrix is best understood visually. The `visualize_report()` method
# creates a comprehensive plot that summarizes the evaluation results.
#
# **How to read the confusion matrix:**
# - Each row represents the *actual* ground truth species.
# - Each column represents the species that the *model predicted*.
# - The diagonal (from top-left to bottom-right) shows the number of correct
#   predictions for each class.
# - Off-diagonal numbers indicate misclassifications. For example, a number
#   in row "A" and column "B" means an image of species A was incorrectly
#   classified as species B.

# %%
# Pass the report dictionary to the visualization function
classifier.visualize_report(report)

# %% [markdown]
# ## 7. Batch Prediction for Efficiency
#
# If your goal is to classify many images using `predict_batch()` is much more efficient than looping
# over `predict()`, if it leverages the GPU to process images in parallel,
# results will be returned in a significant speed-up.

# %%
# We'll use the same small batch from our evaluation example
print(
    f"\n--- Classifying a batch of {len(batch_images)} images with predict_batch ---",
)
batch_predictions = classifier.predict_batch(batch_images, show_progress=True)

print("\n--- Batch Classification Results (Top prediction for each image) ---")
for i, single_image_preds in enumerate(batch_predictions):
    if single_image_preds:  # Check if the prediction list is not empty
        top_pred_species = single_image_preds[0][0]
        top_pred_conf = single_image_preds[0][1]
        print(
            f"  - Image {i+1} (GT: {ground_truths[i]}): "
            f"Predicted '{top_pred_species}' with {top_pred_conf:.2%} confidence.",
        )
    else:
        print(f"  - Image {i+1} (GT: {ground_truths[i]}): Prediction failed.")
