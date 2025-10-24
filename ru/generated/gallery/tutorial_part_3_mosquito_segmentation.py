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
# Руководство по сегментации комаров

Это руководство демонстрирует, как использовать библиотеку `culicidaelab`
для выполнения сегментации комаров на изображениях. Мы рассмотрим:

1. Настройку модели сегментации
2. Загрузку данных для сегментации из набора данных
3. Выполнение сегментации
4. Визуализацию результатов
5. Оценку производительности с использованием истинных (эталонных) масок

"""

# %% [markdown]
# Установите библиотеку `culicidaelab`, если она еще не установлена
# ```bash
# !pip install -q culicidaelab[full]
# ```
# или, если есть доступ к GPU
# ```bash
# !pip install -q culicidaelab[full-gpu]
# ```
#
# Импортируем необходимые библиотеки

# %%
import matplotlib.pyplot as plt
import numpy as np

from culicidaelab import MosquitoSegmenter, MosquitoDetector
from culicidaelab import DatasetsManager, get_settings

# %% [markdown]
# ## 1. Инициализация настроек и загрузка набора данных
#
# Сначала мы инициализируем наши настройки, создадим `MosquitoSegmenter` и загрузим набор данных для сегментации:

# %%
# Получаем экземпляр настроек и инициализируем менеджер наборов данных
settings = get_settings()
manager = DatasetsManager(settings)

# Загружаем набор данных для сегментации
seg_data = manager.load_dataset("segmentation", split="train[:20]")

# Инициализируем сегментатор и детектор
segmenter = MosquitoSegmenter(settings=settings, load_model=True)
detector = MosquitoDetector(settings=settings, load_model=True)

# %% [markdown]
# ## 2. Изучение образца для сегментации
#
# Давайте изучим образец из набора данных для сегментации, чтобы понять его структуру:

# %%
# Изучаем образец для сегментации
seg_sample = seg_data[0]
seg_image = seg_sample["image"]
seg_mask = np.array(seg_sample["label"])  # Преобразуем маску в массив numpy

print(f"Размер изображения: {seg_image.size}")
print(f"Форма маски сегментации: {seg_mask.shape}")
print(f"Уникальные значения в маске: {np.unique(seg_mask)}")  # 0 - фон, 1 и выше - комар

# Создаем цветное наложение для маски
# Где значения в маске равны 1 и выше (комар), делаем ее красной
overlay = np.zeros((*seg_mask.shape, 4), dtype=np.uint8)
overlay[seg_mask >= 1] = [255, 0, 0, 128]  # Красный цвет с 50% прозрачностью

# %% [markdown]
# ## 3. Запуск сегментации на изображении из набора данных
#
# Теперь мы можем запустить модель сегментации на нашем изображении из набора данных:

# %%
# Запускаем детекцию для получения ограничивающих рамок
result = detector.predict(seg_image)
bboxes = [detection.box.to_numpy() for detection in result.detections]

# Запускаем сегментацию с рамками детекции
predicted_mask = segmenter.predict(seg_image, detection_boxes=np.array(bboxes))

# Создаем визуализации
annotated_image = detector.visualize(seg_image, result)
segmented_image = segmenter.visualize(annotated_image, predicted_mask)

# %% [markdown]
# ## 4. Визуализация результатов со сравнением с истинной маской
#
# Давайте визуализируем результаты сегментации рядом с истинной (эталонной) маской:

# %%
plt.figure(figsize=(20, 10))

# Исходное изображение
plt.subplot(2, 4, 1)
plt.imshow(seg_image)
plt.axis("off")
plt.title("Исходное изображение")

# Истинная маска
plt.subplot(2, 4, 2)
plt.imshow(seg_mask, cmap="gray")
plt.axis("off")
plt.title("Истинная маска")

# Наложение истинной маски
plt.subplot(2, 4, 3)
plt.imshow(seg_image)
plt.imshow(overlay, alpha=0.5)
plt.axis("off")
plt.title("Наложение истинной маски")

# Детекции
plt.subplot(2, 4, 4)
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Обнаруженные комары")

# Предсказанная маска
plt.subplot(2, 4, 5)
plt.imshow(predicted_mask.mask, cmap="gray")
plt.axis("off")
plt.title("Предсказанная маска")

# Наложение предсказанной маски
predicted_overlay = np.zeros((*predicted_mask.mask.shape, 4), dtype=np.uint8)
predicted_overlay[predicted_mask.mask >= 0.5] = [0, 255, 0, 128]  # Зеленый для предсказаний
plt.subplot(2, 4, 6)
plt.imshow(seg_image)
plt.imshow(predicted_overlay, alpha=0.5)
plt.axis("off")
plt.title("Наложение предсказанной маски")

# Комбинированное наложение (истинная маска + предсказания)
combined_overlay = np.zeros((*predicted_mask.mask.shape, 4), dtype=np.uint8)
combined_overlay[seg_mask >= 1] = [255, 0, 0, 128]  # Красный для истинной маски
combined_overlay[predicted_mask.mask >= 0.5] = [0, 255, 0, 128]  # Зеленый для предсказаний
plt.subplot(2, 4, 7)
plt.imshow(seg_image)
plt.imshow(combined_overlay, alpha=0.5)
plt.axis("off")
plt.title("Комбинированное наложение\n(Красный: Истинная, Зеленый: Предск.)")

# Конечное сегментированное изображение
plt.subplot(2, 4, 8)
plt.imshow(segmented_image)
plt.axis("off")
plt.title("Конечное сегментированное изображение")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Оценка качества сегментации
#
# Давайте оценим результаты сегментации, используя истинную маску:

# %%
metrics = segmenter.evaluate(
    prediction=predicted_mask,
    ground_truth=seg_mask,
)
print("Метрики оценки сегментации:")
for key, value in metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")
