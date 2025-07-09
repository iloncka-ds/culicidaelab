# Быстрый старт

Этот гайд покажет, как за несколько шагов загрузить изображение, классифицировать вид комара и визуализировать результат.

### 1. Импорт и инициализация

Сначала импортируем необходимые классы и получаем объект настроек.

```python
import cv2
from pathlib import Path
from culicidae_lab import get_settings
from culicidae_lab.predictors import MosquitoClassifier
from culicidae_lab.utils import download_file # Для загрузки тестового изображения

# Получаем стандартные настройки
settings = get_settings()

# Инициализируем классификатор. Модель будет скачана и загружена при необходимости.
classifier = MosquitoClassifier(settings=settings, load_model=True)
```
### 2. Подготовка данных

Загрузим тестовое изображение.

```python
# Загрузим пример изображения
image_url = "URL_К_ВАШЕМУ_ТЕСТОВОМУ_ИЗОБРАЖЕНИЮ.jpg" # Замените на реальный URL
image_path = download_file(image_url, downloads_dir=settings.cache_dir)

# Чтение изображения с помощью OpenCV
image_bgr = cv2.imread(str(image_path))
# Конвертация в RGB, т.к. большинство моделей ожидают этот формат
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
```

### 3. Получение предсказания

```python
# Предсказание вида
predictions = classifier.predict(image_rgb)

print("Результаты классификации:")
for species, confidence in predictions:
    print(f"- {species}: {confidence:.4f}")
```

### 4. Визуализация

```python
# Визуализируем результат на исходном изображении
output_image = classifier.visualize(image_rgb, predictions)

# Сохраняем результат
save_path = Path("./result.png")
cv2.imwrite(str(save_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

print(f"Визуализация сохранена в {save_path}")
```
