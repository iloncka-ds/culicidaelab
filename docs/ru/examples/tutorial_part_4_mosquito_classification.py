"""
# Классификация видов комаров

В этом руководстве показано, как использовать `MosquitoClassifier` из библиотеки CulicidaeLab
для выполнения классификации видов. Мы рассмотрим:

- Загрузка модели классификации
- Подготовка изображения
- Запуск модели для получения результатов классификации
- Визуализация и интерпретация результатов
- Оценка предсказания классификации

"""

# %%
# Установите библиотеку `culicidaelab`, если она еще не установлена
# !pip install -q culicidaelab

# %% [markdown]
# ## 1. Инициализация
#
# Мы начнем с инициализации `MosquitoClassifier`. Объект `settings` будет управлять
# конфигурацией, а `load_model=True` обеспечит загрузку модели и
# ее немедленное размещение в памяти.

# %%
import cv2
import re
import matplotlib.pyplot as plt
from pathlib import Path

from culicidaelab import MosquitoClassifier, get_settings

# Получить экземпляр настроек
settings = get_settings()

# Создать экземпляр классификатора и загрузить модель
print("Инициализация MosquitoClassifier и загрузка модели...")
classifier = MosquitoClassifier(settings, load_model=True)
print("Модель успешно загружена.")

# Вы можете проверить конфигурацию модели непосредственно из объекта настроек.
classifier_conf = settings.get_config("predictors.classifier")
print(f"\nЗагруженная архитектура модели: {classifier_conf.model_arch}")


# %% [markdown]
# ### Проверка классов модели
# Часто бывает полезно посмотреть, какие виды модель была обучена распознавать.
# Рекомендуемый способ — использовать `species_config`.

# %%
species_map = settings.species_config.species_map
print("--- Классы модели (из настроек) ---")
print(species_map)

# %% [markdown]
# ## 2. Подготовка изображения для классификации
#
# Теперь давайте загрузим тестовое изображение. Классификатор ожидает изображение в формате RGB.

# %%
# Загрузить тестовое изображение
image_path = Path("test_imgs") / "640px-Aedes_aegypti.jpg"
image = cv2.imread(str(image_path))
if image is None:
    raise ValueError(f"Не удалось загрузить изображение из {image_path}")

# Преобразовать из BGR (по умолчанию в OpenCV) в RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Отобразить входное изображение
plt.figure(figsize=(8, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Входное изображение")
plt.show()

# %% [markdown]
# ## 3. Запуск классификации и интерпретация результатов
#
# Метод `predict` возвращает список кортежей, где каждый кортеж содержит
# `(имя_вида, оценка_уверенности)`, отсортированный по уверенности.

# %%
# Запустить классификацию
predictions = classifier.predict(image_rgb)

# Давайте выведем топ-5 предсказаний
print("--- Топ-5 предсказаний ---")
for species, prob in predictions[:5]:
    print(f"{species}: {prob:.2%}")

# %% [markdown]
# ## 4. Визуализация результатов классификации
#
# `CulicidaeLab` предоставляет два простых способа визуализации результатов:
# 1. Гистограмма, показывающая уверенность для всех классов.
# 2. Аннотированное изображение с наложенными лучшими предсказаниями.

# %%
# Получить все названия видов и их предсказанные вероятности
species_names = [p[0] for p in predictions]
probabilities = [p[1] for p in predictions]
conf_threshold = settings.get_config("predictors.classifier.confidence")

# Создать гистограмму вероятностей
plt.figure(figsize=(12, 7))
bars = plt.barh(species_names[::-1], probabilities[::-1])  # Перевернуть, чтобы показать самые высокие значения сверху

# Окрасить столбцы в зависимости от уверенности
for i, prob in enumerate(probabilities[::-1]):
    bars[i].set_color("teal" if prob >= conf_threshold else "lightgray")

plt.axvline(x=conf_threshold, color="r", linestyle="--", label=f"Порог уверенности ({conf_threshold:.0%})")
plt.xlabel("Вероятность")
plt.title("Вероятности классификации видов")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Метод `visualize` накладывает лучшие предсказания непосредственно на изображение.
annotated_image = classifier.visualize(image_rgb, predictions)

# Отобразить аннотированное изображение
plt.figure(figsize=(10, 10))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Результат классификации с наложением")
plt.show()


# %% [markdown]
# ## 5. Оценка предсказания
#
# Для измерения производительности вы можете оценить предсказание по известной эталонной метке (ground truth).
# Это возвращает ключевые метрики, такие как точность и правильность в топ-5.

# %%
# Истинная метка для нашего тестового изображения — 'aedes_aegypti'
ground_truth_label = "aedes_aegypti"

# Вы можете выполнить оценку на основе предварительно вычисленного предсказания...
print(f"--- Оценка предсказания по эталонной метке '{ground_truth_label}' ---")
metrics_from_prediction = classifier.evaluate(ground_truth_label, prediction=predictions)
print(f"Метрики из предсказания: {metrics_from_prediction}")

# ... или непосредственно из входного изображения.
metrics_from_image = classifier.evaluate(ground_truth_label, input_data=image_rgb)
print(f"Метрики из изображения: {metrics_from_image}")

# %% [markdown]
# ## 6. Пакетная классификация
#
# Так же, как и детектор, классификатор может обрабатывать пакет изображений для повышения производительности.

# %%
image_dir = Path("test_imgs")
pattern = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)
image_paths = [path for path in image_dir.iterdir() if path.is_file() and pattern.search(str(path))]

try:
    batch = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in image_paths]
    print(f"\n--- Классификация пакета из {len(batch)} изображений ---")
except Exception as e:
    print(f"Произошла ошибка при чтении изображений: {e}")
    batch = []

classifier_batch_result = classifier.predict_batch(batch, show_progress=True)
print("\n--- Результаты пакетной классификации ---")
for i, single_image_preds in enumerate(classifier_batch_result):
    top_pred_species = single_image_preds[0][0]
    top_pred_conf = single_image_preds[0][1]
    print(
        f"  - Изображение '{image_paths[i].name}': ",
        f"Лучшее предсказание — '{top_pred_species}' с уверенностью {top_pred_conf:.2%}.",
    )
