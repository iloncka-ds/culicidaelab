# Руководство пользователя CulicidaeLab

Добро пожаловать в Руководство пользователя `CulicidaeLab`! Это руководство проведет вас через основные функции библиотеки, показывая, как выполнять обнаружение, классификацию и сегментацию комаров. Мы рассмотрим все, от начальной настройки до выполнения предсказаний и визуализации результатов.

## 1. Прежде чем начать: основные концепции

### Объект `settings`

Единственная наиболее важная концепция в `CulicidaeLab` — это **объект `settings`**. Это ваша централизованная точка входа ко всему. Вместо того чтобы импортировать и инициализировать каждый компонент вручную со сложными параметрами, вы просто:

1.  Получаете объект `settings`.
2.  Просите его создать нужный вам компонент (`Detector`, `Classifier` и т. д.).

Объект `settings` автоматически обрабатывает загрузку конфигураций, управление путями к файлам, загрузку моделей и выбор оптимального бэкенда для вашего случая использования.

### Рабочий процесс предиктора

Все три основных компонента (`MosquitoDetector`, `MosquitoClassifier`, `MosquitoSegmenter`) являются **предикторами**. Они имеют общий, последовательный и предсказуемый рабочий процесс:

1.  **Инициализация**: Создать экземпляр с помощью упрощенного конструктора, который автоматически выбирает лучший бэкенд.
2.  **Загрузка модели**: Веса модели загружаются лениво, то есть они загружаются и помещаются в память только при первом предсказании или когда вы явно укажете это сделать. Для ясности мы будем использовать `load_model=True`.
3.  **Предсказание**: Использовать метод `.predict()` на изображении для получения структурированных, типобезопасных результатов.
4.  **Визуализация**: Использовать метод `.visualize()` для просмотра результатов.

### Структурированные результаты предсказаний

CulicidaeLab теперь возвращает **структурированные результаты предсказаний** вместо простых кортежей или списков. Это означает:

- **Типобезопасность**: Все выходные данные являются валидированными моделями Pydantic с четкой структурой
- **Легкий доступ**: Используйте удобные методы, такие как `.top_prediction()` для результатов классификации
- **Богатая информация**: Каждое предсказание включает оценки уверенности, ограничивающие рамки и метаданные
- **JSON-сериализуемость**: Идеально подходит для веб-API и хранения данных

### Автоматический выбор бэкенда

Библиотека автоматически выбирает лучший бэкенд для ваших нужд:

- **Режим разработки**: Использует PyTorch для полной гибкости и возможностей отладки
- **Производственный режим**: Автоматически выбирает ONNX для оптимальной производительности и меньшего объема памяти
- **Прозрачность**: Один и тот же API предсказаний работает независимо от бэкенда

Давайте начнем!

## 2. Инициализация и настройка

Сначала давайте настроим нашу среду и получим всемогущий объект `settings`.

```python
# Импорт необходимых библиотек для обработки изображений и построения графиков
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Импорт основных компонентов CulicidaeLab
from culicidaelab import get_settings, MosquitoDetector, MosquitoClassifier, MosquitoSegmenter

# Получение центрального объекта настроек.
# Этот единственный объект будет использоваться для инициализации всех остальных компонентов.
settings = get_settings()

print("Настройки CulicidaeLab успешно инициализированы.")
```

## 3. Обнаружение комаров

Первым шагом во многих конвейерах анализа является выяснение, *есть ли* комар на изображении и *где* он находится. Это работа `MosquitoDetector`.

### 3.1. Инициализация детектора и загрузка изображения

Когда мы инициализируем детектор с `load_model=True`, библиотека проверяет, присутствуют ли веса модели YOLO локально. Если нет, они будут автоматически загружены и кэшированы для всех будущих использований.

```python
# Инициализация детектора.
# С load_model=True веса модели будут загружены в память.
print("Инициализация MosquitoDetector...")
detector = MosquitoDetector(settings=settings, load_model=True)
print("Модель детектора загружена и готова.")

# Давайте загрузим тестовое изображение для работы.
# Убедитесь, что вы заменили этот путь на путь к вашему собственному изображению.
image_path = Path("test_imgs") / "640px-Aedes_aegypti.jpg"

# CulicidaeLab принимает различные форматы изображений:
# - Путь к файлу (str или Path)
# - PIL Image (уже в RGB)
# - NumPy массив
# - Байты изображения
# Для простоты мы используем PIL Image, который уже в RGB режиме
image = Image.open(image_path)
image_rgb = np.array(image)
```

### 3.2. Выполнение обнаружения

Метод `predict` принимает изображение в качестве входных данных и возвращает список всех обнаруженных объектов. Каждый объект представляет собой кортеж, содержащий координаты ограничивающей рамки и оценку уверенности. Формат: `(center_x, center_y, width, height, confidence)`.

```python
# Запуск предсказания на нашем RGB-изображении
result = detector.predict(image_rgb)

# Давайте выведем результаты в удобочитаемом формате.
print("\nРезультаты обнаружения:")
if result.detections:
    for i, detection in enumerate(result.detections):
        bbox = detection.box
        conf = detection.confidence
        print(
            f"  - Комар {i+1}: Уверенность = {conf:.2f}, "
            f"Рамка = (x1={bbox.x1:.1f}, y1={bbox.y1:.1f}, "
            f"x2={bbox.x2:.1f}, y2={bbox.y2:.1f})"
        )
else:
    print("  На изображении комары не обнаружены.")
```

### 3.3. Визуализация результатов обнаружения

Чтение координат полезно, но видеть результат лучше. Метод `.visualize()` рисует ограничивающие рамки и оценки уверенности непосредственно на изображении.

```python
# Передача исходного изображения и результата обнаружения в метод visualize
annotated_image = detector.visualize(image_rgb, result)

# Использование matplotlib для отображения итогового изображения
plt.figure(figsize=(10, 7))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("Обнаруженные комары")
plt.show()
```

## 4. Классификация видов комаров

Как только у вас есть изображение комара, следующий вопрос часто звучит так: "Какой это вид?" `MosquitoClassifier` обучен отвечать на этот вопрос.

### 4.1. Инициализация классификатора

Как и в случае с детектором, мы инициализируем классификатор из объекта `settings`. Соответствующая модель классификации будет загружена при первом использовании.

```python
# Инициализация классификатора
print("\nИнициализация MosquitoClassifier...")
classifier = MosquitoClassifier(settings=settings, load_model=True)
print("Модель классификатора загружена и готова.")

# Для этого примера мы будем использовать то же изображение, что и для обнаружения.
# В реальном приложении вы могли бы использовать обрезанное изображение с выхода детектора.
```

### 4.2. Выполнение классификации

Метод `predict` классификатора возвращает список всех возможных видов, отсортированный по оценке уверенности. Каждый элемент — это кортеж `(название_вида, оценка_уверенности)`.

```python
# Запуск классификации
result = classifier.predict(image_rgb)

# Вывод топ предсказания с помощью удобного метода
top_pred = result.top_prediction()
if top_pred:
    print(f"\nТоп предсказание: {top_pred.species_name} ({top_pred.confidence:.4f})")

# Вывод трех наиболее вероятных видов
print("\nТоп-3 предсказания:")
for prediction in result.predictions[:3]:
    print(f"- {prediction.species_name}: {prediction.confidence:.4f}")
```

### 4.3. Интерпретация и визуализация результатов классификации

Гистограмма — отличный способ понять уверенность модели по всем потенциальным видам. Давайте визуализируем топ-5 предсказаний.

```python
# Извлечение названий и вероятностей топ-5 предсказаний для нашего графика
top_5_predictions = result.predictions[:5]
species_names = [pred.species_name for pred in top_5_predictions]
probabilities = [pred.confidence for pred in top_5_predictions]

# Создание горизонтальной гистограммы
plt.figure(figsize=(10, 6))
plt.barh(species_names, probabilities, color="skyblue")
plt.xlabel("Вероятность")
plt.title("Вероятности классификации видов (Топ-5)")
plt.gca().invert_yaxis()  # Инвертирование оси для отображения наиболее вероятного результата сверху

# Добавление значений вероятностей в виде текста на столбцах для наглядности
for index, value in enumerate(probabilities):
    plt.text(value, index, f" {value:.2%}", va='center')

plt.tight_layout()
plt.show()
```

## 5. Сегментация комаров

Сегментация идет на шаг дальше, чем обнаружение. Вместо простой рамки она предоставляет точную, попиксельную маску, очерчивающую точную форму комара.

### 5.1. Инициализация сегментатора

Снова мы инициализируем наш `MosquitoSegmenter` из объекта `settings`.

```python
# Инициализация сегментатора
print("\nИнициализация MosquitoSegmenter...")
segmenter = MosquitoSegmenter(settings=settings, load_model=True)
print("Модель сегментатора загружена и готова.")
```

### 5.2. Выполнение сегментации

Метод `predict` возвращает бинарную маску (2D numpy массив). В этой маске пиксели, принадлежащие комару, помечены как `True` (или `255`), а пиксели фона — как `False` (или `0`).

Существует два способа выполнения сегментации:

#### **Метод 1: Базовая сегментация**

Вы можете запустить сегментатор на всем изображении. Он попытается найти и сегментировать самый заметный объект.

```python
print("\n--- Выполнение базовой сегментации на всем изображении ---")
basic_result = segmenter.predict(image_rgb)
basic_mask = basic_result.mask
print("Базовая сегментация завершена.")
```

#### **Метод 2: Сегментация с использованием результатов обнаружения (рекомендуется)**

Для наилучших результатов вы можете передать ограничивающие рамки, полученные от `MosquitoDetector`. Это говорит сегментатору, где именно искать, что приводит к более точной и чистой маске.

```python
# Мы будем использовать результаты обнаружения от детектора ранее
# Преобразование обнаружений в формат, ожидаемый сегментатором
detection_boxes = [detection.box.to_numpy() for detection in result.detections]

print("\n--- Выполнение сегментации с использованием рамок обнаружения в качестве ориентира ---")
guided_result = segmenter.predict(image_rgb, detection_boxes=detection_boxes)
guided_mask = guided_result.mask
print("Сегментация с ориентиром завершена.")
```

### 5.3. Визуализация результатов сегментации

Метод `.visualize()` идеально подходит для просмотра итоговой маски, наложенной на исходное изображение.

```python
# Визуализация более точного, управляемого результата сегментации
segmented_image = segmenter.visualize(image_rgb, guided_result)

# Отображение исходного изображения, самой маски и итогового наложения
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.axis("off")
plt.title("Исходное изображение")

plt.subplot(1, 3, 2)
plt.imshow(guided_mask, cmap="gray")
plt.axis("off")
plt.title("Маска сегментации (с ориентиром)")

plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.axis("off")
plt.title("Наложение сегментации")

plt.tight_layout()
plt.show()
```

## 6. Продвинутые функции и производственное развертывание

### 6.1. Высокопроизводительный Serve API

Для производственных приложений и сценариев с высокой пропускной способностью CulicidaeLab предоставляет специальный API `serve`, который обеспечивает значительные улучшения производительности благодаря автоматическому выбору бэкенда ONNX и интеллектуальному кэшированию.

#### Базовое использование Serve API

```python
from culicidaelab.serve import serve, clear_serve_cache

# Serve API принимает те же форматы изображений, что и обычные предикторы:
# - Пути к файлам (str или Path): "image.jpg"
# - PIL Images: Image.open("image.jpg")
# - NumPy массивы: np.array(image)
# - Байты изображения: image_bytes

# Быстрая классификация
result = serve("mosquito.jpg", predictor_type="classifier")
species = result.top_prediction().species_name
confidence = result.top_prediction().confidence

print(f"Обнаружен вид: {species} с уверенностью {confidence:.2%}")

# Быстрое обнаружение
result = serve("image.jpg", predictor_type="detector", confidence_threshold=0.7)
detections = result.detections
print(f"Обнаружено {len(detections)} комаров")

# Быстрая сегментация
result = serve("image.jpg", predictor_type="segmenter")
mask = result.mask
print(f"Создана маска размером {mask.shape}")

# Очистка кэша по завершении
clear_serve_cache()
```

#### Сравнение производительности

```python
import time
from culicidaelab import MosquitoClassifier, get_settings
from culicidaelab.serve import serve, clear_serve_cache

settings = get_settings()

# Традиционный подход (бэкенд PyTorch)
classifier = MosquitoClassifier(settings, load_model=True)
start = time.time()
result1 = classifier.predict("image.jpg")
torch_time = time.time() - start

# Serve API (бэкенд ONNX с кэшированием)
start = time.time()
result2 = serve("image.jpg", predictor_type="classifier")  # Первый вызов - загружает модель
serve_time_first = time.time() - start

start = time.time()
result3 = serve("image.jpg", predictor_type="classifier")  # Последующий вызов - использует кэш
serve_time_cached = time.time() - start

print(f"PyTorch: {torch_time:.3f}с")
print(f"Serve (первый): {serve_time_first:.3f}с")
print(f"Serve (кэшированный): {serve_time_cached:.3f}с")  # Обычно в 10-100 раз быстрее

clear_serve_cache()
```

### 6.2. Утилитарные функции для обнаружения

CulicidaeLab предоставляет утилитарные функции для программного обнаружения доступных моделей и наборов данных:

```python
from culicidaelab import list_models, list_datasets

# Получение списка всех доступных моделей
available_models = list_models()
print("Доступные модели:")
for model in available_models:
    print(f"  - {model}")

# Получение списка всех доступных наборов данных
available_datasets = list_datasets()
print("\nДоступные наборы данных:")
for dataset in available_datasets:
    print(f"  - {dataset}")
```

### 6.3. Управление памятью и контекстные менеджеры

Для эффективной обработки памяти используйте контекстные менеджеры для автоматической загрузки и выгрузки моделей:

```python
from culicidaelab import MosquitoClassifier, get_settings

settings = get_settings()
classifier = MosquitoClassifier(settings)

# Временная загрузка модели для эффективности памяти
with classifier.model_context():
    predictions = classifier.predict(image)
    # Модель автоматически выгружается после контекста
    
print("Модель выгружена, память освобождена")
```

### 6.4. Пакетная обработка и оценка

Для продвинутых пользователей, которым необходимо обрабатывать большие наборы данных или оценивать производительность модели, каждый предиктор поддерживает улучшенные пакетные операции:

```python
# Пакетная обработка с отслеживанием прогресса
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = classifier.predict_batch(
    input_data_batch=images,
    show_progress=True
)

# Оценка производительности модели
from pathlib import Path

# Подготовка данных для оценки
test_images = list(Path("test_data").glob("*.jpg"))
ground_truths = ["Aedes_aegypti", "Anopheles_gambiae", "Culex_pipiens"]

# Запуск оценки
evaluation_report = classifier.evaluate_batch(
    input_data_batch=test_images,
    ground_truth_batch=ground_truths,
    show_progress=True
)

print("Результаты оценки:")
for metric, value in evaluation_report.items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.4f}")
    else:
        print(f"  {metric}: {value}")
```

### 6.5. Лучшие практики производственного развертывания

#### Интеграция с веб-API

```python
from fastapi import FastAPI, UploadFile
from culicidaelab.serve import serve
import json

app = FastAPI()

@app.post("/predict/{predictor_type}")
async def predict(predictor_type: str, file: UploadFile):
    # Чтение загруженного изображения
    image_bytes = await file.read()
    
    # Запуск предсказания
    result = serve(image_bytes, predictor_type=predictor_type)
    
    # Возврат JSON-ответа
    return json.loads(result.model_dump_json())

@app.on_event("shutdown")
async def shutdown():
    clear_serve_cache()
```

#### Пакетная обработка с кэшированием

```python
from culicidaelab.serve import serve, clear_serve_cache

# Эффективная обработка множества изображений
images = ["img1.jpg", "img2.jpg", "img3.jpg"]

results = []
for image in images:
    # Первый вызов загружает модель, последующие используют кэшированную модель
    result = serve(image, predictor_type="classifier")
    results.append(result)

# Очистка после пакетной обработки
clear_serve_cache()
```

Эти продвинутые функции делают CulicidaeLab подходящим как для исследовательских задач, так и для производственных развертываний с высокими требованиями к производительности.
