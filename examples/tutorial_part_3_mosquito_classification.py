"""
Сегментация комаров с помощью MosquitoSegmenter
===============================================

Этот туториал показывает, как использовать `MosquitoSegmenter` из библиотеки CulicidaeLab
для выполнения сегментации комаров на изображениях. Мы рассмотрим:

- Загрузку модели сегментатора (SAM)
- Подготовку изображения
- Запуск предсказания для получения маски
- Визуализацию результата

"""
# %% [markdown]
# # Mosquito Classification Tutorial
# This tutorial demonstrates how to use CulicidaeLab for mosquito species classification.
# First, let's import the necessary libraries:
# 1. Setting up the classification model
# 2. Loading and preprocessing images
# 3. Running classification
# 4. Interpreting results

# %%
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from culicidaelab import MosquitoClassifier
from culicidaelab.core.settings import get_settings
from pathlib import Path

# %% [markdown]
# ## 1. Initialize Settings and Classifier
#

# %% [markdown]
# First, we'll initialize our settings and create a MosquitoClassifier instance.
# The settings module will handle downloading model weights if they're not already present:

# %%
# Get settings instance
settings = get_settings()
settings.list_model_types()

# %%
classifier_conf = settings.get_config("predictors.classifier")
print(classifier_conf.model_arch)

# %%
classifier_conf.model_dump()

# %%
print(classifier_conf.repository_id)
print(classifier_conf.filename)

# %%
# Get model path and config
model_path = settings.get_model_weights_path("classifier")
model_config = settings.get_config("predictors.classifier").model_dump()
print(f"Using model path: {model_path}")
print(f"Using model config: {model_config}")

# %%
classifier = MosquitoClassifier(settings, load_model=True)

# %%
classes_sp = classifier.learner.dls.vocab

# %%
species_map = {i: s for i, s in enumerate(classes_sp)}
print(species_map)

# %% [markdown]
# ## 2. Load and Preprocess Image
#
# Now let's load a test image. We'll use one of the sample images provided in the test_data directory:

# %%
# Find test image path
image_path = Path("test_imgs") / "640px-Aedes_aegypti.jpg"

# Load and preprocess image
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Could not load image from {image_path}")

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.title("Input Image")
plt.show()

# %% [markdown]
# ## 3. Run Classification
#
# Let's classify the mosquito species. The classifier will return probabilities for each species class:

# %%
# Run classification
predictions = classifier.predict(image)

# Get species names and probabilities from predictions
species_names = [p[0] for p in predictions]  # First element of each tuple is species name
probabilities = [p[1] for p in predictions]  # Second element is the confidence score

# Sort predictions by probability
sorted_indices = np.argsort(probabilities)[::-1]
sorted_species = [species_names[i] for i in sorted_indices]
sorted_probs = [probabilities[i] for i in sorted_indices]

# Print top predictions based on config
top_k = model_config["params"].get("top_k", 3)
conf_threshold = model_config["params"].get("conf_threshold", 0.5)

print(f"\nTop {top_k} predictions (confidence threshold: {conf_threshold:.0%}):")
for species, prob in zip(sorted_species[:top_k], sorted_probs[:top_k]):
    if prob >= conf_threshold:
        print(f"{species}: {prob:.1%}")
    else:
        print(f"{species}: {prob:.1%} (below threshold)")

# %% [markdown]
# ## 4. Visualize Results
#
# Let's create a bar plot of the classification probabilities:

# %%
# Plot probabilities
plt.figure(figsize=(12, 6))
bars = plt.bar(sorted_species, sorted_probs)

# Color bars based on confidence threshold
for i, prob in enumerate(sorted_probs):
    bars[i].set_color("green" if prob >= conf_threshold else "gray")

plt.axhline(y=conf_threshold, color="r", linestyle="--", label="Confidence Threshold")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Probability")
plt.title("Species Classification Probabilities")
plt.legend()
plt.tight_layout()
plt.show()

# %%
annotated_image = classifier.visualize(
    image,
    predictions,
    save_path="annotated_image.jpg",
)

plt.figure(figsize=(15, 5))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Top 5 predictions")

# %%
metrics = classifier.evaluate("Aedes albopictus", input_data=image)
print(metrics)

# %%
metrics = classifier.evaluate("Aedes albopictus", predictions)
print(metrics)

# %%
image_dir = Path("test_imgs")

# This pattern matches any string that ends with .jpg, .jpeg, or .png, case-insensitively.
# \.   -> matches a literal dot
# (jpg|jpeg|png) -> matches 'jpg' OR 'jpeg' OR 'png'
# $    -> matches the end of the string
pattern = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)

# Get list with all files and filter using the regex
image_paths = [path for path in image_dir.iterdir() if path.is_file() and pattern.search(str(path))]
try:
    batch = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in image_paths]
    print(f"\nSuccessfully created a batch with {len(batch)} images.")
except Exception as e:
    print(f"An error occurred while reading images: {e}")
    batch = []

classifier_batch_result = classifier.predict_batch(batch)
print(classifier_batch_result)
