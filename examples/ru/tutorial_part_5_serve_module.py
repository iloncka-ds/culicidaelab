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
# Использование модуля serve для инференса в production

Это руководство демонстрирует, как использовать функцию `serve` из библиотеки
CulicidaeLab для высокопроизводительной работы на устройствах с минимальными
ресурсами.
Опция `serve` разработана как легковесный, быстрый и безопасный способ
выполнения предсказаний.

Это руководство охватывает:

- **Скорость и безопасность**: Как функция `serve` использует бэкенд ONNX для быстрого инференса.
- **Предсказание для одного изображения**: Как использовать `serve` для задачи классификации.
- **Кэширование**: Понимание механизма кэширования в памяти для экземпляров предикторов.
- **Очистка кэша**: Как очистить кэш при необходимости.
"""

# %% [markdown]
# Установите библиотеку `culicidaelab`, если она еще не установлена
# ```bash
# !pip install -q culicidaelab
# ```
#
#
# ## 1. Инициализация и настройка
#
# Мы инициализируем `DatasetsManager`, чтобы получить некоторые демонстрационные данные.
# Функция `serve` не требует ручной инициализации предикторов.

# %%
# Импорт необходимых библиотек
import matplotlib.pyplot as plt

# Импорт необходимых классов из библиотеки CulicidaeLab
from culicidaelab import (
    DatasetsManager,
    get_settings,
    serve,
    clear_serve_cache,
)

# Получение экземпляра настроек библиотеки по умолчанию
settings = get_settings()

# Инициализация сервисов, необходимых для управления и загрузки данных
manager = DatasetsManager(settings)

# %% [markdown]
# ## 2. Загрузка тестового набора данных
#
# Мы будем использовать встроенный тестовый набор данных, чтобы получить изображение
# для наших предсказаний.

# %%
print("\n--- Загрузка тестового сплита набора данных 'classification' ---")
classification_test_data = manager.load_dataset("classification", split="test")
print("Тестовый набор данных успешно загружен!")
print(f"Количество образцов в тестовом наборе данных: {len(classification_test_data)}")

# Давайте выберем один образец для работы.
classification_test_data = classification_test_data.shuffle(seed=42)
sample = classification_test_data[0]
image = sample["image"]
ground_truth_label = sample["label"]

print(f"\nИстинная метка выбранного образца: '{ground_truth_label}'")

# Отображение входного изображения
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Входное изображение\n(Истинная метка: {ground_truth_label})")
plt.axis("off")
plt.show()


# %% [markdown]
# ## 3. Использование `serve` для классификации
#
# Функция `serve` автоматически инициализирует предиктор с бэкендом ONNX
# при первом вызове и кэширует его для последующих запросов.

# %%
# Запуск классификации с использованием функции serve
print("--- Запуск классификации в первый раз (произойдет инициализация предиктора) ---")
classification_result = serve(image, predictor_type="classifier")

# Вывод топ-5 предсказаний
print("\n--- Топ-5 предсказаний классификации ---")
for p in classification_result.predictions[:5]:
    print(f"{p.species_name}: {p.confidence:.2%}")

# %% [markdown]
# ## 4. Кэширование в действии
#
# Если вы снова выполните тот же запрос, вы заметите, что он выполняется
# намного быстрее, потому что предиктор уже находится в памяти.

# %%
# Повторный запуск классификации для демонстрации эффекта кэширования
print("\n--- Повторный запуск классификации (должно быть быстрее) ---")
classification_result_cached = serve(image, predictor_type="classifier")

# Повторный вывод топ-5 предсказаний
print("\n--- Топ-5 предсказаний классификации (из кэша) ---")
for p in classification_result_cached.predictions[:5]:
    print(f"{p.species_name}: {p.confidence:.2%}")


# %% [markdown]
# ## 5. Очистка кэша
#
# Если вам нужно освободить память или перезагрузить предикторы, вы можете
# использовать функцию `clear_serve_cache`.

# %%
# Очистка кэша
print("\n--- Очистка кэша предикторов ---")
clear_serve_cache()

# Повторный запуск классификации, предиктор будет повторно инициализирован
print("\n--- Повторный запуск классификации после очистки кэша (произойдет повторная инициализация) ---")
classification_result_after_clear = serve(image, predictor_type="classifier")

print("\n--- Топ-5 предсказаний классификации (после очистки кэша) ---")
for p in classification_result_after_clear.predictions[:5]:
    print(f"{p.species_name}: {p.confidence:.2%}")
