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
# # Mosquito Detection Tutorial
#
# This notebook demonstrates how to use the CulicidaeLab library for detecting mosquitoes in images.
#

# %%
import re
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from culicidaelab.core.settings import get_settings
from culicidaelab.predictors.detector import MosquitoDetector


# %% [markdown]
# ## 1. Initialize Settings and Load Model
#
# First, we'll get the settings instance which will handle model weights and configurations.

# %%
# Get settings instance
settings = get_settings()
settings.list_model_types()

# %%
model_config = settings.get_config("predictors.detector")
model_path = settings.get_model_weights_path("detector")

# %%
detector = MosquitoDetector(settings=settings, load_model=True)

# %% [markdown]
# ## 2. Load and Process an Image
#
# Now let's load a test image and run detection on it.

# %%
# Load test image
image_path = str(Path("test_imgs") / "640px-Aedes_aegypti.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run detection
detections = detector.predict(image)

# Draw detections
annotated_image = detector.visualize(image, detections)

# Display results
plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Detected Mosquitoes")
plt.show()

# Print detection results
print("\nDetection Results:")
for i, (x, y, w, h, conf) in enumerate(detections):
    print(f"Mosquito {i+1}: Confidence = {conf:.2f}, Box = (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})")

# %%
test_ground_truth = detections[0][:4]
evaluation = detector.evaluate([test_ground_truth], prediction=detections)
print(evaluation)
evaluation_from_raw = detector.evaluate([test_ground_truth], input_data=image)
print(evaluation_from_raw)

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

detections_batch = detector.predict_batch(batch)
print(detections_batch)

# %%
# Assuming detections_batch is in the format: [[(x,y,w,h,conf), ...], [(x,y,w,h,conf), ...], ...]
# Create ground truth batch in the correct format
batch_test_gt = [[(x, y, w, h) for (x, y, w, h, conf) in detections] for detections in detections_batch]

# Now call evaluate_batch with the correct format
batch_evaluation = detector.evaluate_batch(
    ground_truth_batch=batch_test_gt,  # List of lists of ground truth boxes
    input_data_batch=None,  # We're providing predictions directly
    predictions_batch=detections_batch,  # List of lists of predictions with confidence
    num_workers=1,  # Use single worker for deterministic results
)

print(batch_evaluation)
