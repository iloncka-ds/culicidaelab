"""
# Руководство по сегментации комаров

В этом руководстве демонстрируется, как использовать библиотеку CulicidaeLab
для выполнения сегментации комаров на изображениях. Мы рассмотрим:

1. Настройка модели сегментации
2. Загрузка и предварительная обработка изображений
3. Запуск сегментации
4. Визуализация результатов

"""

# %%
# Установите библиотеку `culicidaelab`, если она еще не установлена
# !pip install -q culicidaelab

# %%
# Сначала давайте импортируем необходимые библиотеки:
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from culicidaelab import MosquitoSegmenter, MosquitoDetector
from culicidaelab import ModelWeightsManager
from culicidaelab import get_settings

# %% [markdown]
# ## 1. Инициализация настроек и сегментатора
#
# Сначала мы инициализируем наши настройки и создадим экземпляр MosquitoSegmenter:

# %%
# Получить экземпляр настроек
settings = get_settings()
settings.list_model_types()

# %%
model_config = settings.get_config("predictors.segmenter")
model_path = settings.get_model_weights_path("segmenter")

weights_manager = ModelWeightsManager(settings=settings)
# Инициализация сегментатора
segmenter = MosquitoSegmenter(settings=settings, load_model=True)

# %% [markdown]
# ## 2. Загрузка и предварительная обработка изображения
#
# Теперь давайте загрузим тестовое изображение:

# %%
# Загрузка тестового изображения
image_path = str(Path("test_imgs") / "640px-Aedes_aegypti.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# %% [markdown]
# ## 3. Запуск сегментации
#
# Теперь мы можем запустить модель сегментации на нашем изображении:

# %%
mask = segmenter.predict(image)

# %% [markdown]
# ## 4. Визуализация результатов
#
# Наконец, давайте визуализируем результаты сегментации, наложенные на исходное изображение:

# %%
# Визуализация результатов сегментации
segmented_image = segmenter.visualize(image, mask)

# Отображение результатов
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Исходное изображение")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.axis("off")
plt.title("Маска сегментации")

plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.axis("off")
plt.title("Сегментированное изображение")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Сегментация с использованием результатов обнаружения
#

# %% [markdown]
# Сегментатор также может использовать результаты обнаружения для повышения точности сегментации.
# Вот как объединить обнаружение и сегментацию:

# %%
# Инициализация детектора
detector = MosquitoDetector(settings=settings, load_model=True)
# Загрузка тестового изображения
image_path = str(Path("test_imgs") / "640px-Aedes_aegypti.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Запуск обнаружения
detections = detector.predict(image)

# Запуск сегментации с рамками обнаружения
mask_with_boxes = segmenter.predict(image, detection_boxes=detections)
# Отрисовка обнаружений
annotated_image = detector.visualize(image, detections)

# Вывод результатов обнаружения
print("\nРезультаты обнаружения:")
for i, (x, y, w, h, conf) in enumerate(detections):
    print(
        f"Комар {i+1}: Уверенность = {conf:.2f}, Рамка = (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})",
    )
# Визуализация результатов
segmented_image_with_boxes = segmenter.visualize(annotated_image, mask_with_boxes)

# plt.figure(figsize=(10, 10))
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(mask_with_boxes, cmap="gray")
plt.axis("off")
plt.title("Маска сегментации")


plt.subplot(1, 2, 2)
plt.imshow(segmented_image_with_boxes)
plt.axis("off")
plt.title("Сегментация с рамками обнаружения")

plt.tight_layout()
plt.show()

# %%
metrics = segmenter.evaluate(mask_with_boxes, input_data=image)
print(metrics)

# %%
metrics_default = segmenter.evaluate(
    mask_with_boxes,
    mask_with_boxes,
)
print(metrics_default)
