"""
# Использование модуля `settings`

В этом руководстве демонстрируется, как использовать основной объект `settings` в CulicidaeLab.
Объект `settings` является основной точкой входа для доступа к конфигурациям, путям к файлам
и параметрам моделей во всей библиотеке.
"""
# %%
# Установите библиотеку `culicidaelab`, если она еще не установлена
# !pip install -q culicidaelab

# %%
import yaml
from pathlib import Path

# %%
from culicidaelab import get_settings

# %% [markdown]
# ## 1. Использование настроек по умолчанию
#
# Самый простой способ начать работу с `CulicidaeLab` — это загрузить настройки по умолчанию.
# Функция `get_settings()` действует как синглтон; она загружает конфигурацию один раз
# и возвращает тот же экземпляр при последующих вызовах. Это обеспечивает согласованное
# состояние во всем вашем приложении.
#
# Настройки по умолчанию загружаются из файлов конфигурации, поставляемых с библиотекой.

# %%
# Получить экземпляр настроек по умолчанию
settings = get_settings()

# Объект настроек обеспечивает легкий доступ к ключевым каталогам ресурсов.
# Библиотека автоматически создаст эти каталоги, если они не существуют.
print("--- Каталоги ресурсов по умолчанию ---")
print(f"Активный каталог конфигурации: {settings.config_dir}")
print(f"Каталог моделей: {settings.model_dir}")
print(f"Каталог наборов данных: {settings.dataset_dir}")
print(f"Каталог кэша: {settings.cache_dir}")

# %% [markdown]
# ## 2. Доступ к путям весов моделей
#
# Объект `settings` знает локальные пути по умолчанию для всех весов моделей-предикторов.
# Когда вы создаете экземпляр предиктора, он использует эти пути для поиска или загрузки моделей.

# %%
# Получить настроенные локальные пути к файлам для разных типов моделей
detection_weights = settings.get_model_weights_path("detector")
segmentation_weights = settings.get_model_weights_path("segmenter")
classification_weights = settings.get_model_weights_path("classifier")

print("--- Пути к весам моделей по умолчанию ---")
print(f"Модель обнаружения: {detection_weights}")
print(f"Модель сегментации: {segmentation_weights}")
print(f"Модель классификации: {classification_weights}")

# %% [markdown]
# ## 3. Работа с конфигурацией видов
#
# Вся информация, связанная с видами, включая названия классов и подробные метаданные,
# управляется через свойство `species_config`. Это крайне важно для интерпретации
# выходных данных модели классификации.

# %%
# Получить специальный объект конфигурации видов
species_config = settings.species_config

# Вы можете легко получить сопоставление индексов классов с названиями видов.
print("\n--- Сопоставление индексов видов с названиями ---")
for idx, species in species_config.species_map.items():
    print(f"Класс {idx}: {species}")

# Вы также можете получить подробные метаданные для любого конкретного вида.
species_name = "Aedes aegypti"
metadata = species_config.get_species_metadata(species_name)
if isinstance(metadata, dict):
    print(f"\n--- Метаданные для '{species_name}' ---")
    for key, value in metadata.items():
        print(f"{key}: {value}")

# %% [markdown]
# ## 4. Использование пользовательского каталога конфигурации
#
# Для более сложных случаев использования, таких как предоставление собственных метаданных о видах или изменение
# параметров модели по умолчанию, вы можете указать библиотеке на пользовательский каталог конфигурации.
#
# `CulicidaeLab` загрузит ваши пользовательские файлы `.yaml` и объединит их с настройками по умолчанию.
# Это позволяет вам переопределить только те настройки, которые необходимо изменить.

# %%
# Создать пользовательский каталог конфигурации и новый файл конфигурации
custom_config_dir = Path("custom_configs")
custom_config_dir.mkdir(exist_ok=True)

# Определим минимальную пользовательскую конфигурацию. Мы просто переопределим информацию о видах.
# Все настройки, не определенные здесь, будут использовать значения по умолчанию из библиотеки.
example_config = {
    "species": {
        "species_classes": {0: "Aedes aegypti", 1: "Anopheles gambiae"},
        "species_metadata": {
            "species_info_mapping": {
                "aedes_aegypti": "Aedes aegypti",
                "anopheles_gambiae": "Anopheles gambiae",
            },
            "species_metadata": {
                "Aedes aegypti": {
                    "common_name": "Пользовательский комар-переносчик желтой лихорадки",
                    "taxonomy": {
                        "family": "Culicidae",
                        "subfamily": "Culicinae",
                        "genus": "Aedes",
                    },
                    "metadata": {
                        "vector_status": True,
                        "diseases": ["Dengue", "Zika"],
                        "habitat": "Urban",
                        "breeding_sites": ["Artificial containers"],
                        "sources": ["custom_source"],
                    },
                },
                "Anopheles gambiae": {
                    "common_name": "Пользовательский африканский малярийный комар",
                    "taxonomy": {
                        "family": "Culicidae",
                        "subfamily": "Anophelinae",
                        "genus": "Anopheles",
                    },
                    "metadata": {
                        "vector_status": True,
                        "diseases": ["Malaria"],
                        "habitat": "Rural",
                        "breeding_sites": ["Puddles"],
                        "sources": ["custom_source"],
                    },
                },
            },
        },
    },
}


# Записать пользовательский файл конфигурации
config_file_path = custom_config_dir / "species.yaml"
with open(config_file_path, "w") as f:
    yaml.safe_dump(example_config, f)

# Теперь инициализируем настройки с путем к нашему пользовательскому каталогу.
# `get_settings` достаточно умен, чтобы создать *новый* экземпляр, если указан другой `config_dir`.
print("\n--- Инициализация с пользовательскими настройками ---")
custom_settings = get_settings(config_dir=str(custom_config_dir))

print(f"Активный каталог конфигурации: {custom_settings.config_dir}")

# Давайте проверим, загрузилась ли наша пользовательская карта видов
print("\n--- Пользовательское сопоставление видов ---")
for idx, species in custom_settings.species_config.species_map.items():
    print(f"Класс {idx}: {species}")

# %% [markdown]
# ## 5. Переопределение одного значения конфигурации
#
# Иногда вам может потребоваться изменить только одно значение во время выполнения без создания новых YAML-файлов.
# Метод `set_config` идеально подходит для этого.
#
# Давайте загрузим настройки по умолчанию и изменим порог уверенности для детектора.

# %%
# Снова загрузить настройки по умолчанию (или использовать предыдущий экземпляр 'settings')
runtime_settings = get_settings()
original_confidence = runtime_settings.get_config("predictors.detector.confidence")
print(f"Исходная уверенность детектора: {original_confidence}")

# Установить новое значение порога уверенности детектора с помощью `set_config`
runtime_settings.set_config("predictors.detector.confidence", 0.85)
new_confidence = runtime_settings.get_config("predictors.detector.confidence")
print(f"Новая уверенность детектора: {new_confidence}")
