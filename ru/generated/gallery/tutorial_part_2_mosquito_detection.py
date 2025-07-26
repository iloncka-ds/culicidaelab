"""
# Руководство по обнаружению комаров

В этом руководстве показано, как использовать `MosquitoDetector` из библиотеки CulicidaeLab
для выполнения обнаружения объектов на изображениях. Мы рассмотрим:

- Загрузка модели детектора
- Подготовка изображения
- Запуск модели для получения ограничивающих рамок
- Визуализация результатов
- Оценка точности предсказания
- Выполнение предсказаний на пакете изображений

"""
# %% [markdown]
# ## 1. Инициализация
#
# Сначала мы получим глобальный экземпляр `settings` и используем его для инициализации нашего `MosquitoDetector`.
# Устанавливая `load_model=True`, мы указываем детектору немедленно загрузить веса модели в память.
# Если файл модели не существует локально, он будет загружен автоматически.

# %%
import re
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from culicidaelab import get_settings
from culicidaelab import MosquitoDetector

# Получить экземпляр настроек
settings = get_settings()

# Инициализировать детектор и загрузить модель
print("Инициализация MosquitoDetector и загрузка модели...")
detector = MosquitoDetector(settings=settings, load_model=True)
print("Модель успешно загружена.")

# %% [markdown]
# ## 2. Обнаружение комаров на одном изображении
#
# Теперь загрузим тестовое изображение и запустим на нем детектор.

# %%
# Загрузка тестового изображения из локального каталога 'test_imgs'
image_path = Path("test_imgs") / "640px-Aedes_aegypti.jpg"
image = cv2.imread(str(image_path))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Преобразование в RGB для matplotlib

# Метод `predict` возвращает список обнаружений.
# Каждое обнаружение — это кортеж: (center_x, center_y, width, height, confidence_score)
detections = detector.predict(image_rgb)

# Метод `visualize` рисует ограничивающие рамки на изображении для удобного просмотра.
annotated_image = detector.visualize(image_rgb, detections)

# Отображение результата
plt.figure(figsize=(12, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Обнаруженные комары")
plt.show()

# Вывод численных результатов обнаружения
print("\nРезультаты обнаружения:")
if detections:
    for i, (x, y, w, h, conf) in enumerate(detections):
        print(f"  - Комар {i+1}: Уверенность = {conf:.2f}, Рамка = (x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f})")
else:
    print("  Комары не обнаружены.")

# %% [markdown]
# ## 3. Оценка предсказания
#
# Метод `evaluate` позволяет сравнить предсказание с эталонными данными (ground truth).
# Это полезно для измерения точности модели. Метод возвращает несколько метрик,
# включая среднюю точность (Average Precision, AP), которая является стандартом для обнаружения объектов.
#
# Здесь мы будем использовать только что найденное обнаружение в качестве имитации эталонных данных,
# чтобы продемонстрировать процесс.

# %%
# Эталонные данные (ground truth) — это список рамок без оценки уверенности: [(x, y, w, h), ...]
if detections:
    test_ground_truth = [detections[0][:4]]  # Использовать первую обнаруженную рамку в качестве наших эталонных данных

    # Вы можете выполнить оценку, используя предварительно вычисленное предсказание
    print("--- Оценка с использованием предварительно вычисленного предсказания ---")
    evaluation = detector.evaluate(ground_truth=test_ground_truth, prediction=detections)
    print(evaluation)

    # Или вы можете позволить методу выполнить предсказание внутренне, передав необработанное изображение
    print("\n--- Оценка непосредственно из изображения ---")
    evaluation_from_raw = detector.evaluate(ground_truth=test_ground_truth, input_data=image_rgb)
    print(evaluation_from_raw)
else:
    print("Пропуск оценки, так как обнаружения не найдены.")


# %% [markdown]
# ## 4. Выполнение пакетных предсказаний
#
# Для эффективности вы можете обрабатывать несколько изображений одновременно, используя `predict_batch`.
# Это намного быстрее, чем перебирать в цикле и вызывать `predict` для каждого изображения по отдельности.

# %%
# Найти все файлы изображений в каталоге 'test_imgs'
image_dir = Path("test_imgs")
pattern = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)
image_paths = [path for path in image_dir.iterdir() if path.is_file() and pattern.search(str(path))]

# Загрузить все изображения в список (наш "пакет")
try:
    batch = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in image_paths]
    print(f"\n--- Обработка пакета из {len(batch)} изображений ---")
except Exception as e:
    print(f"Произошла ошибка при чтении изображений: {e}")
    batch = []

# Запустить пакетное предсказание
detections_batch = detector.predict_batch(batch)
print("Пакетное предсказание завершено.")
for i, dets in enumerate(detections_batch):
    print(f"  - Изображение {i+1} ({image_paths[i].name}): Найдено {len(dets)} обнаружение(й).")


# %% [markdown]
# ## 5. Оценка пакета предсказаний
#
# Аналогично, `evaluate_batch` можно использовать для получения агрегированных метрик по всему набору изображений.

# %%
# Создать имитацию пакета эталонных данных из результатов нашего пакетного предсказания
batch_test_gt = [[(x, y, w, h) for (x, y, w, h, conf) in detections] for detections in detections_batch]

# Вызвать evaluate_batch. Мы предоставляем предсказания напрямую.
print("\n--- Оценка всего пакета ---")
batch_evaluation = detector.evaluate_batch(
    ground_truth_batch=batch_test_gt,
    predictions_batch=detections_batch,
    num_workers=1,
)

print("Агрегированные метрики оценки пакета:")
print(batch_evaluation)
