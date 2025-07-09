**CulicidaeLab** — это мощная и гибкая Python-библиотека для анализа изображений комаров.

Она предоставляет готовые решения для:

- **Детекции**: обнаружение комаров на изображении.
- **Сегментации**: выделение точных масок комаров.
- **Классификации**: определение вида комара.

### Ключевые возможности

- **Готовые модели**: Используйте предобученные модели для быстрого старта.
- **Гибкая конфигурация**: Управляйте всеми аспектами через YAML-файлы.
- **Оценка качества**: Встроенные инструменты для оценки точности моделей.
- **Расширяемость**: Легко добавляйте свои модели и источники данных.

### Пример использования

```python
import cv2
from culicidaelab import get_settings
from culicidaelab.predictors import MosquitoClassifier

# Получаем настройки по умолчанию
settings = get_settings()

# Инициализируем классификатор (модель загрузится при первом вызове)
classifier = MosquitoClassifier(settings=settings)

# Загружаем ваше изображение (например, с помощью OpenCV)

image = cv2.imread("path/to/your/mosquito_image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Получаем предсказание
predictions = classifier.predict(image_rgb)

# Выводим лучший результат
top_prediction = predictions[0]
print(f"Вид: {top_prediction[0]}, Точность: {top_prediction[1]:.2f}")
```
