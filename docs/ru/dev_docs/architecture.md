# Руководство разработчика
## 1. Введение

### 1.1 Назначение
Этот документ предоставляет детальное описание проекта для библиотеки CulicidaeLab. Библиотека представляет собой систему на Python, предназначенную для облегчения управления наборами данных, конфигурацией и моделями машинного обучения (predictors) для задач, связанных с анализом изображений комаров, включая детекцию, сегментацию и классификацию.

### 1.2 Область применения
Библиотека предоставляет:
-   Типобезопасную, валидированную систему конфигурации для параметров моделей, настроек приложения и метаданных видов.
-   Кросс-платформенное управление ресурсами для наборов данных, весов моделей, кэша и временных файлов.
-   Набор моделей (predictors) для классификации, детекции и сегментации, построенных на общем базовом классе.
-   Архитектуру на основе провайдеров для интеграции с внешними источниками данных, такими как Hugging Face.
-   Утилиты для обработки наборов данных, файловых операций и визуализации результатов.

### 1.3 Определения, акронимы и сокращения (DAA)
-   **YOLO**: You Only Look Once (архитектура модели детекции объектов)
-   **SAM**: Segment Anything Model (архитектура модели сегментации)
-   **IoU**: Intersection over Union (метрика для сегментации и детекции)
-   **AP**: Average Precision (метрика для детекции)
-   **mAP**: Mean Average Precision (метрика детекции)
-   **Pydantic**: Библиотека валидации данных и управления настройками для Python.

### 1.4 Ссылки
*   IEEE Std 1016-1998, Рекомендуемая практика IEEE для описаний проектирования ПО.
*   Справочник по языку Python 3.x.
*   Документация Pydantic.
*   Документация OmegaConf.
*   Документация библиотеки FastAI.
*   Документация HuggingFace Hub и Datasets.
*   Документация модели YOLO (Ultralytics).
*   Документация Segment Anything Model (SAM).
*   Документация timm (PyTorch Image Models).

### 1.5 Обзор документа
Этот документ организован в четыре основных раздела:
*   **Раздел 1 (Введение)**: Предоставляет назначение, область применения, определения, ссылки и обзор этого документа.
*   **Раздел 2 (Обзор системы)**: Описывает контекст системы, цели и общие функции.
*   **Раздел 3 (Архитектурное проектирование системы)**: Описывает высокоуровневую архитектуру, включая декомпозицию компонентов, их интерфейсы и проектирование данных.
*  **Раздел 4 (Детальное проектирование системы)**: Предоставляет детальное описание каждого модуля и его составных классов, включая их назначение, функции, интерфейсы и данные.
*  **Раздел 5 (Прослеживаемость)**: Кратко обсуждает, как проектирование отвечает неявным требованиям библиотеки.

---

## 2. Обзор системы

### 2.1 Контекст системы
Библиотека CulicidaeLab предназначена для использования исследователями, разработчиками и специалистами по данным, работающими над проектами, связанными с анализом изображений комаров. Она может служить бэкендом для более крупных приложений или использоваться напрямую в скриптах для пакетной обработки, оценки моделей и управления наборами данных. Она работает в среде Python и зависит от нескольких внешних библиотек для своей основной функциональности (например, Pydantic, FastAI, библиотеки Hugging Face, PyTorch, Ultralytics).

### 2.2 Цели системы
Основные цели этой библиотеки:
1.  Предоставить централизованный, типобезопасный и валидированный способ управления конфигурациями для наборов данных, моделей и настроек приложений с использованием Pydantic.
2.  Предложить гибкую систему на основе провайдеров для управления и доступа к различным наборам данных и весам моделей из различных источников (например, Hugging Face).
3.  Определить общий, предсказуемый интерфейс (`BasePredictor`) для различных типов моделей (детектор, сегментатор, классификатор) и предоставить конкретные реализации.
4.  Обеспечить эффективный вывод моделей, четкую визуализацию результатов и стандартизированную оценку.
5.  Гарантировать надежное, кросс-платформенное управление ресурсами для моделей, наборов данных, кэша и временных файлов.

### 2.3 Функции системы
Библиотека предоставляет следующие ключевые функции:
*   **Управление конфигурацией**: Загружает, объединяет и валидирует иерархические конфигурации из YAML-файлов в модели Pydantic.
*   **Управление ресурсами**: Управляет путями файловой системы для моделей, наборов данных, кэша и временных файлов кросс-платформенным способом.
*   **Абстракция провайдеров**: Загружает наборы данных и веса моделей от внешних провайдеров (например, Hugging Face) через стандартизированный интерфейс.
*   **Управление наборами данных**: Перечисляет и загружает наборы данных, как определено в конфигурации, используя абстракцию провайдеров.
*   **Предсказание модели**: Выполняет вывод с использованием моделей детектора, сегментатора и классификатора на отдельных изображениях или батчах.
*   **Визуализация результатов**: Генерирует визуальные выходы предсказаний моделей, наложенные на входные изображения.
*   **Оценка модели**: Оценивает производительность модели против эталонных данных с использованием стандартных метрик для каждой задачи.
*   **Управление весами моделей**: Гарантирует, что веса моделей доступны локально, загружая их через провайдеров при необходимости.

---

## 3. Архитектурное проектирование системы
### 3.1 Архитектурный обзор и принципы проектирования
Архитектура библиотеки `culicidaelab` построена на принципах Clean Code и SOLID, чтобы предоставить пользователям мощный и гибкий инструмент, остающийся простым в использовании. Архитектура четко разделена на логические слои, каждый со своей областью ответственности. Это разделение упрощает понимание, тестирование и расширение кода.

#### 3.1.1 Диаграмма архитектурных слоев
```mermaid
graph TD
    %% -- Layers Definition --
    classDef userLayer fill:#eaf4ff,stroke:#004085,stroke-width:2px,color:#004085
    classDef productLayer fill:#e8f5e9,stroke:#155724,stroke-width:2px,color:#155724
    classDef facadeLayer fill:#fff8e1,stroke:#856404,stroke-width:2px,color:#856404
    classDef coreLayer fill:#fbe9e7,stroke:#721c24,stroke-width:2px,color:#721c24
    classDef infraLayer fill:#eceff1,stroke:#383d41,stroke-width:2px,color:#383d41
    classDef externalLayer fill:#f5f5f5,stroke:#6c757d,stroke-width:4px,stroke-dasharray: 5 5

    %% -- User Layer --
    subgraph " "
        direction LR
        subgraph "Пользователь"
            User[/"👤 Пользователь"/]
        end
    end

    %% -- Products Layer --
    subgraph "Основной функционал библиотеки"
        direction TB


        Predictors["🔬 Predictors"]
        style Predictors fill:#fff,stroke:none
        MosquitoClassifier("🦟 MosquitoClassifier")
        MosquitoDetector("🎯 MosquitoDetector")
        MosquitoSegmenter("✂️ MosquitoSegmenter")
    end

    %% -- Facade Layer --
    subgraph "Фасад (Центр Конфигурации)"
        facade_spacer["<br>"]; style facade_spacer fill:none,stroke:none
        facade_spacer ~~~ Settings

        Settings["🎛️ Settings<br/><b>(Фасад)</b><br/>Предоставляет унифицированный доступ<br/>к сервисам и параметрам"]
    end

    %% -- Core Abstraction & Logic Layer --
    subgraph "Ядро (Абстракции и Контракты)"
        core_spacer["<br>"]; style core_spacer fill:none,stroke:none
        core_spacer ~~~ BasePredictor

        BasePredictor["🧩 BasePredictor<br/>Абстрактный класс для<br/>всех предикторов"]
        BaseProvider["🔌 BaseProvider<br/>Абстрактный класс для<br/>внешних провайдеров"]
        WeightsManagerProtocol["⚖️ WeightsManagerProtocol<br/>Интерфейс для управления весами"]
        ProviderService["🔧 ProviderService<br/>Фабрика для создания<br/>объектов-провайдеров"]
        ConfigManager["⚙️ ConfigManager<br/>Управляет конфигурациями<br/>из YAML файлов"]
    end

    %% -- Infrastructure & Adapters Layer --
    subgraph "Инфраструктура и Адаптеры"
        infra_spacer["<br>"]; style infra_spacer fill:none,stroke:none
        infra_spacer ~~~ DatasetsManager

        ModelWeightsManager["⚖️ ModelWeightsManager<br/>Реализация протокола<br/>управления весами"]
        ResourceManager["📁 ResourceManager<br/>Управляет файлами<br/>и директориями"]
        HuggingFaceProvider["🤗 HuggingFaceProvider<br/><b>(Адаптер)</b><br/>Реализация для<br/>Hugging Face Hub"]
        DatasetsManager["📊 DatasetsManager<br/>Управляет жизненным циклом<br/>наборов данных"]
    end

    %% -- External Systems --
    subgraph "Внешние Системы"
        direction LR


        HuggingFaceHub[("🌐<br/>Hugging Face Hub")]
        FileSystem[("💾<br/>Локальная<br/>Файловая Система")]
    end

    %% -- Relationships --
    User -- "Использует" --> MosquitoClassifier
    User -- "Использует" --> MosquitoDetector
    User -- "Использует" --> MosquitoSegmenter
    User -- "Или управляет данными через" --> DatasetsManager

    MosquitoClassifier -- "Получает при создании" --> Settings
    MosquitoDetector -- "Получает при создании" --> Settings
    MosquitoSegmenter -- "Получает при создании" --> Settings

    Settings -- "Управляет" --> ProviderService
    Settings -- "Управляет" --> ConfigManager
    Settings -- "Управляет" --> ResourceManager

    MosquitoClassifier -. "Реализует" .-> BasePredictor
    MosquitoDetector -. "Реализует" .-> BasePredictor
    MosquitoSegmenter -. "Реализует" .-> BasePredictor
    Predictors --- MosquitoClassifier
    Predictors --- MosquitoDetector
    Predictors --- MosquitoSegmenter

    BasePredictor -- "Требует" --> WeightsManagerProtocol
    ModelWeightsManager -. "Реализует" .-> WeightsManagerProtocol
    ModelWeightsManager -- "Использует" --> ProviderService

    DatasetsManager -- "Использует" --> ProviderService
    ProviderService -- "Создает" --> HuggingFaceProvider
    HuggingFaceProvider -. "Реализует" .-> BaseProvider

    HuggingFaceProvider -- "Загружает из" --> HuggingFaceHub
    ResourceManager -- "Работает с" --> FileSystem
    HuggingFaceProvider -- "Сохраняет в" --> ResourceManager
    ConfigManager -- "Читает из" --> ResourceManager

    %% -- Styling --
    class User userLayer
    class MosquitoClassifier,MosquitoDetector,MosquitoSegmenter,Predictors productLayer
    class Settings facadeLayer
    class BasePredictor,BaseProvider,WeightsManagerProtocol,ProviderService,ConfigManager coreLayer
    class ModelWeightsManager,ResourceManager,HuggingFaceProvider,DatasetsManager infraLayer
    class HuggingFaceHub,FileSystem externalLayer
```

#### 3.1.2 Описание слоев
1.  **Основная функциональность библиотеки**: Это высший уровень, представляющий конкретные, используемые продукты библиотеки: `MosquitoClassifier`, `MosquitoDetector` и `MosquitoSegmenter`.
2.  **Фасад (Центр конфигурации)**: Класс `Settings` реализует паттерн проектирования **Фасад**. Он служит единой, упрощенной точкой входа для настройки всей библиотеки, скрывая внутреннюю сложность управления конфигурациями, ресурсами и сервисами.
3.  **Ядро (Абстракции и контракты)**: Это архитектурное ядро, определяющее основные абстрактные классы и интерфейсы (`BasePredictor`, `BaseProvider`, `WeightsManagerProtocol`). Этот слой полностью отделен от конкретных реализаций.
4.  **Инфраструктура и адаптеры**: Этот слой содержит конкретные реализации базовых абстракций. Он действует как мост между логикой библиотеки и внешним миром.
    *   `ModelWeightsManager` и `DatasetsManager` управляют высокоуровневыми ресурсами.
    *   `HuggingFaceProvider` реализует паттерн **Адаптер**, адаптируя API Hugging Face к внутреннему интерфейсу `BaseProvider`.
    *   `ResourceManager` работает напрямую с файловой системой.
5.  **Внешние системы**: Ресурсы вне прямого контроля библиотеки, такие как `Hugging Face Hub` и локальная `Файловая система`.

#### 3.1.3 Руководящие принципы проектирования
*   **Расширяемость**: Чтобы добавить новый источник данных (например, AWS S3), разработчику нужно только создать новый `S3Provider`, реализующий интерфейс `BaseProvider`, и зарегистрировать его в конфигурации. Изменения в высокоуровневых модулях моделей не требуются.
*   **Поддерживаемость и тестируемость**: Единственная ответственность каждого компонента упрощает отладку. Использование инверсии зависимостей позволяет заменять инфраструктурные компоненты на моки во время тестирования.
*   **Принципы SOLID**:
    *   **Принцип инверсии зависимостей (DIP)**: Высокоуровневые модули (`MosquitoClassifier`) не зависят от низкоуровневых модулей (`HuggingFaceProvider`). Оба зависят от абстракций (`BaseProvider`).
    *   **Паттерн Фабрика (`ProviderService`)**: Позволяет системе динамически решать, какой объект провайдера создавать на основе конфигурации.
    *   **Паттерн Фасад (`Settings`)**: Упрощает взаимодействие пользователя с библиотекой, скрывая сложность создания и связывания внутренних компонентов.
### 3.2 Декомпозиция компонентов
Библиотека декомпозирована на четыре основных высокоуровневых модуля:

1.  **Модуль `core`**:
    *   **Описание**: Формирует основу библиотеки, предоставляя основные сервисы, базовые классы, протоколы и модели данных.
    *   **Подкомпоненты**: `Settings`, `ConfigManager`, `ResourceManager`, `BasePredictor`, `BaseProvider`, `ProviderService`, `WeightsManagerProtocol`, `config_models`, `species_config`, `utils`.
2.  **Модуль `datasets`**:
    *   **Описание**: Обрабатывает высокоуровневую логику для управления и доступа к наборам данных.
    *   **Подкомпоненты**: `DatasetsManager`.
3.  **Модуль `providers`**:
    *   **Описание**: Содержит конкретные реализации `core.BaseProvider` для получения данных из различных внешних источников.
    *   **Подкомпоненты**: `HuggingFaceProvider`.
4.  **Модуль `predictors`**:
    *   **Описание**: Содержит конкретные реализации `BasePredictor` для конкретных задач машинного обучения и конкретную реализацию `WeightsManagerProtocol`.
    *   **Подкомпоненты**: `MosquitoClassifier`, `MosquitoDetector`, `MosquitoSegmenter`, `ModelWeightsManager`.

### 3.3 Интерфейсы компонентов

*   **`core.Settings`**: Основной класс, с которым взаимодействует пользователь, доступный через `get_settings()`. Действует как синглтон-фасад, предоставляя простой доступ к значениям конфигурации, путям к ресурсам (`.model_dir`) и вспомогательным объектам. Инициализирует и хранит экземпляры `ConfigManager` и `ResourceManager`.
*   **`core.ConfigManager`**: Внутренний компонент, управляемый `Settings`. Загружает YAML-файлы по умолчанию и пользовательские, объединяет их и валидирует результат по отношению к моделям Pydantic, определенным в `core.config_models`.
*   **`core.ResourceManager`**: Предоставляет стандартизированные пути для хранения данных (модели, наборы данных, кэш), используемые `Settings`, `DatasetsManager` и `ModelWeightsManager`.
*   **`core.BaseProvider`**: Абстрактный базовый класс, определяющий контракт для любого компонента, предоставляющего данные (наборы данных или файлы моделей), с методами типа `download_dataset` и `download_model_weights`.
*   **`core.ProviderService`**: Фабрика и кэш для экземпляров провайдеров. Использует `Settings` для поиска конфигурации провайдера и создает правильную реализацию `BaseProvider` (например, `providers.HuggingFaceProvider`).
*   **`datasets.DatasetsManager`**: Управляет доступом к наборам данных. Использует `Settings` для получения конфигураций наборов данных и `ProviderService` для получения правильного экземпляра провайдера для загрузки и загрузки данных.
*   **`core.WeightsManagerProtocol`**: Протокол, определяющий метод `ensure_weights`. Это отделяет модели от конкретной реализации управления весами.
*   **`predictors.ModelWeightsManager`**: Конкретная реализация `WeightsManagerProtocol`. Использует `Settings` для поиска конфигураций моделей и `ProviderService` для загрузки файлов весов, если они недоступны локально.
*   **`core.BasePredictor`**: Абстрактный базовый класс для всех моделей. Определяет стандартный интерфейс (`predict`, `evaluate`, `visualize`). Требует объект `Settings` и объект, соответствующий `WeightsManagerProtocol` для инициализации.
*   **`predictors.*` (например, `MosquitoClassifier`, `MosquitoDetector`)**: Конкретные реализации `BasePredictor`. Они инициализируются с `Settings` и экземпляром `ModelWeightsManager`, чтобы гарантировать доступность весов их моделей перед использованием.

### 3.4 Концептуальный поток взаимодействия
```mermaid
sequenceDiagram
    participant User
    participant MosquitoClassifier
    participant ModelWeightsManager
    participant ProviderService
    participant HuggingFaceProvider
    participant ResourceManager
    participant HuggingFaceHub
    participant FastAI

	Note over User, FastAI: Инициализация модели

    User->>MosquitoClassifier: MosquitoClassifier(settings, load_model=True)
    activate MosquitoClassifier
    MosquitoClassifier->>ModelWeightsManager: ensure_weights()
    activate ModelWeightsManager

    ModelWeightsManager->>ProviderService: get_provider("huggingface")
    activate ProviderService
    ProviderService-->>ModelWeightsManager: HuggingFaceProvider instance
    deactivate ProviderService

    ModelWeightsManager->>HuggingFaceProvider: download_model_weights()
    activate HuggingFaceProvider
    HuggingFaceProvider->>ResourceManager: get_save_location()
    activate ResourceManager
    ResourceManager-->>HuggingFaceProvider: Path/to/save
    deactivate ResourceManager
    HuggingFaceProvider->>HuggingFaceHub: Download file
    activate HuggingFaceHub
    HuggingFaceHub-->>HuggingFaceProvider: Model weights file
    deactivate HuggingFaceHub
    HuggingFaceProvider-->>ModelWeightsManager: /path/to/model.pth
    deactivate HuggingFaceProvider
    ModelWeightsManager-->>MosquitoClassifier: /path/to/model.pth
    deactivate ModelWeightsManager
    MosquitoClassifier->>FastAI: load_learner(/path/to/model.pth)
    activate FastAI
    FastAI-->>MosquitoClassifier: learner
    deactivate FastAI
    deactivate MosquitoClassifier

    Note over User, FastAI: Предсказание

    User->>MosquitoClassifier: predict(image)
    activate MosquitoClassifier
    MosquitoClassifier->>FastAI: Perform Inference
    activate FastAI
    FastAI-->>MosquitoClassifier: Predictions
    deactivate FastAI
    MosquitoClassifier-->>User: Predictions
    deactivate MosquitoClassifier

```

Чтобы проиллюстрировать взаимодействие компонентов, рассмотрим типичный сценарий классификации изображения комара.

1.  **Инициализация**: Приложение пользователя вызывает `get_settings()` для получения экземпляра фасада `Settings`. Объект `Settings` загружает все необходимые конфигурации из YAML-файлов. Затем пользователь создает экземпляр `MosquitoClassifier`, передавая ему объект `settings`.
    ```python
    from culicidaelab import MosquitoClassifier, get_settings

    settings = get_settings()
    classifier = MosquitoClassifier(settings=settings, load_model=True)
    ```

2.  **Запрос весов модели**: При инициализации (с `load_model=True`) `MosquitoClassifier` (через своего родителя `BasePredictor`) определяет, что ему нужны веса его модели. Он вызывает метод `ensure_weights` на своем `ModelWeightsManager`.

3.  **Разрешение провайдера**: `ModelWeightsManager` проверяет конфигурацию модели (через `Settings`) и обнаруживает, что веса размещены на Hugging Face. Он запрашивает провайдера "huggingface" у `ProviderService`.

4.  **Создание экземпляра провайдера**: `ProviderService`, действуя как фабрика, создает и возвращает экземпляр `HuggingFaceProvider`.

5.  **Загрузка и сохранение данных**: `ModelWeightsManager` вызывает `download_model_weights` у экземпляра `HuggingFaceProvider`. Провайдер взаимодействует с `Hugging Face Hub`, чтобы загрузить файл. Чтобы определить правильное локальное место сохранения (например, `~/.culicidae_lab/models/...`), он использует `ResourceManager`.

6.  **Загрузка модели**: Проверенный локальный путь к файлу весов возвращается по стеку вызовов в `MosquitoClassifier`. Теперь он может загрузить модель в память (например, модель PyTorch) и быть готовым к выводу.

Весь этот сложный процесс оркестрируется архитектурой и остается скрытым от конечного пользователя, которому нужно выполнить только начальный шаг создания.

### 3.5 Проектирование данных
*   **Данные конфигурации**: Управляются `ConfigManager` и валидируются в дерево **моделей Pydantic**, с `core.config_models.CulicidaeLabConfig` в качестве корня. Исходный источник - YAML-файлы.
*   **Данные изображений**: Представлены как `np.ndarray` (массивы NumPy).
*   **Метаданные набора данных**: Формально определены **моделью Pydantic `core.config_models.DatasetConfig`**.
*   **Предсказания моделей**: Форматы стандартизированы и часто представлены `typing.TypeAlias` для ясности (например, `DetectionPredictionType`).
    *   **Детектор**: `list[tuple[float, float, float, float, float]]` (center_x, center_y, width, height, confidence).
    *   **Сегментатор**: `np.ndarray` (бинарная маска формы HxW).
    *   **Классификатор**: `list[tuple[str, float]]` (species_name, confidence_score).
*   **Эталонные данные**: Аналогично представлены `typing.TypeAlias` (например, `DetectionGroundTruthType`) с форматами, соответствующими типам предсказаний.
*   **Метрики оценки**: Словари, сопоставляющие имена метрик со значениями float (`dict[str, float]`).
*   **Пути файловой системы**: Управляются как объекты `pathlib.Path` через `ResourceManager` и `Settings`.

## 4. Детальное проектирование системы

В этом разделе подробно описывается каждый модуль и его компоненты.

### 4.1 Детальное проектирование модуля `core`
Модуль `core` предоставляет фундаментальные классы, протоколы и утилиты, необходимые для функционирования всей библиотеки. Он обрабатывает конфигурацию, управление ресурсами и определяет базовые контракты для моделей и поставщиков данных.

**Структура проекта:**
```
core
├── __init__.py
├── base_predictor.py
├── base_provider.py
├── config_manager.py
├── config_models.py
├── provider_service.py
├── resource_manager.py
├── settings.py
├── species_config.py
├── utils.py
└── weights_manager_protocol.py
```

#### 4.1.1 `core.base_predictor.BasePredictor`
*   **Идентификация**: `core.base_predictor.BasePredictor`
*   **Назначение**: Абстрактный базовый класс, определяющий общий интерфейс для всех моделей (например, детектора, сегментатора, классификатора).
*   **Наследует**: `Generic[PredictionType, GroundTruthType]`, `ABC`
*   **Функция**: Обеспечивает стандартную структуру для загрузки модели, предсказания, оценки и визуализации. Полагается на основной объект `Settings` для конфигурации и `WeightsManagerProtocol` для управления файлами моделей. Предоставляет контекстный менеджер для временной загрузки модели для эффективного управления памятью.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, predictor_type: str, weights_manager: WeightsManagerProtocol, load_model: bool = False)`: Инициализирует модель.
    *   `load_model(self) -> None`: Обертка, которая загружает модель, если она еще не загружена.
    *   `unload_model(self) -> None`: Выгружает модель для освобождения памяти.
    *   `predict(self, input_data: np.ndarray, **kwargs: Any) -> PredictionType` (abstract): Делает предсказание для одного входа.
    *   `predict_batch(self, input_data_batch: list[np.ndarray], show_progress: bool = True, **kwargs: Any) -> list[PredictionType]`: Делает предсказания для батча входных данных.
    *   `evaluate(self, ground_truth: GroundTruthType, prediction: PredictionType | None = None, input_data: np.ndarray | None = None, **predict_kwargs: Any) -> dict[str, float]`: Оценивает единичное предсказание против эталонных данных.
    *   `evaluate_batch(self, ground_truth_batch: list[GroundTruthType], predictions_batch: list[PredictionType] | None = None, input_data_batch: list[np.ndarray] | None = None, num_workers: int = 4, show_progress: bool = True, **predict_kwargs) -> dict[str, float]`: Оценивает батч элементов с использованием параллельной обработки.
    *   `visualize(self, input_data: np.ndarray, predictions: PredictionType, save_path: str | Path | None = None) -> np.ndarray` (abstract): Визуализирует предсказания на входных данных.
    *   `get_model_info(self) -> dict[str, Any]`: Получает информацию о загруженной модели.
    *   `model_context(self)` (контекстный менеджер): Временно загружает модель для блока кода.
    *   `config` (свойство) `-> PredictorConfig`: Получает модель конфигурации Pydantic для модели.
    *   `model_loaded` (свойство) `-> bool`: Проверяет, загружена ли модель.
    *   `model_path` (свойство) `-> Path`: Получает путь к файлу весов модели.
    *   `__call__(self, input_data: np.ndarray, **kwargs: Any) -> Any`: Удобный метод для `predict()`.
    *   `_load_model(self) -> None` (abstract): Логика загрузки модели, специфичная для дочернего класса.
    *   `_evaluate_from_prediction(self, prediction: PredictionType, ground_truth: GroundTruthType) -> dict[str, float]` (abstract): Основная логика расчета метрик.
*   **Интерфейсы (Используемые)**:
    *   `core.settings.Settings`
    *   `core.weights_manager_protocol.WeightsManagerProtocol`
    *   `core.config_models.PredictorConfig`
*   **Данные**: `settings`, `predictor_type`, `weights_manager`.

#### 4.1.2 `core.base_provider.BaseProvider`
*   **Идентификация**: `core.base_provider.BaseProvider`
*   **Назначение**: Абстрактный базовый класс для всех поставщиков данных и моделей.
*   **Наследует**: `ABC`
*   **Функция**: Определяет стандартный контракт для загрузки наборов данных и весов моделей из внешних или внутренних источников.
*   **Интерфейсы (Предоставляемые)**:
    *   `download_dataset(self, dataset_name: str, save_dir: str | None = None, *args: Any, **kwargs: Any) -> Path` (abstract): Загружает набор данных.
    *   `download_model_weights(self, model_type: str, *args: Any, **kwargs: Any) -> Path` (abstract): Загружает веса модели.
    *   `get_provider_name(self) -> str` (abstract): Получает уникальное имя провайдера.
    *   `load_dataset(self, dataset_path: str | Path, **kwargs: Any) -> Any` (abstract): Загружает набор данных из локального пути.
*   **Данные**: N/A (абстрактный класс).

#### 4.1.3 `core.config_manager.ConfigManager`
*   **Идентификация**: `core.config_manager.ConfigManager`
*   **Назначение**: Обрабатывает загрузку, объединение и валидацию конфигураций для библиотеки.
*   **Функция**: Реализует надежную стратегию загрузки: 1. Загружает конфигурации YAML по умолчанию, входящие в состав библиотеки. 2. Загружает пользовательские конфигурации YAML. 3. Объединяет пользовательскую конфигурацию поверх значений по умолчанию. 4. Валидирует окончательную объединенную конфигурацию по отношению к модели Pydantic `CulicidaeLabConfig`. Также предоставляет утилиту для создания объектов из их определений конфигурации.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, user_config_dir: str | Path | None = None)`: Инициализирует менеджер.
    *   `get_config(self) -> CulicidaeLabConfig`: Возвращает полностью валидированный объект конфигурации Pydantic.
    *   `instantiate_from_config(self, config_obj: Any, **kwargs: Any) -> Any`: Создает объект Python из его модели конфигурации Pydantic, которая должна содержать поле `target`.
    *   `save_config(self, file_path: str | Path) -> None`: Сохраняет текущее состояние конфигурации в YAML-файл.
*   **Интерфейсы (Используемые)**:
    *   `core.config_models.CulicidaeLabConfig` и другие модели Pydantic.
    *   Библиотека `PyYAML` (неявно).
*   **Данные**: `user_config_dir`, `default_config_path`, `config` (экземпляр `CulicidaeLabConfig`).

#### 4.1.4 `core.config_models.py`
*   **Идентификация**: `core.config_models.py`
*   **Назначение**: Определяет модели Pydantic, которые представляют всю структуру конфигурации приложения.
*   **Функция**: Обеспечивает валидацию данных, типобезопасность и четкую структуру для всех разделов конфигурации, гарантируя надежность и предсказуемость.
*   **Ключевые модели (Предоставляемые)**:
    *   `CulicidaeLabConfig`: Корневая модель для всей конфигурации.
    *   `PredictorConfig`: Определяет конфигурацию для одной модели, включая ее класс (`target`), путь к модели и другие параметры.
    *   `ProviderConfig`: Определяет конфигурацию для поставщика данных (например, Hugging Face, Roboflow).
    *   `DatasetConfig`: Определяет конфигурацию для конкретного набора данных.
    *   `SpeciesModel`: Определяет конфигурацию для данных о видах, включая сопоставления классов и метаданные.
    *   `AppSettings`: Основные настройки приложения (например, среда, уровень логирования).
    *   `ProcessingConfig`: Общие параметры обработки (например, размер батча, устройство).
    *   `VisualizationConfig`: Настройки для визуализации выходных данных моделей.

#### 4.1.5 `core.provider_service.ProviderService`
*   **Идентификация**: `core.provider_service.ProviderService`
*   **Назначение**: Управляет созданием экземпляров и жизненным циклом поставщиков данных.
*   **Функция**: Действует как фабрика и кэш для экземпляров провайдеров. Использует `Settings` приложения для поиска конфигурации запрошенного провайдера, создает его с помощью `ConfigManager` и хранит для повторного использования.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings)`: Инициализирует сервис.
    *   `get_provider(self, provider_name: str) -> BaseProvider`: Извлекает созданный экземпляр провайдера по его имени.
*   **Интерфейсы (Используемые)**:
    *   `core.settings.Settings`
    *   `core.base_provider.BaseProvider`
*   **Данные**: `_settings`, `_providers` (как кэш).

#### 4.1.6 `core.resource_manager.ResourceManager`
*   **Идентификация**: `core.resource_manager.ResourceManager`
*   **Назначение**: Централизованное управление ресурсами для моделей, наборов данных и временных файлов с кросс-платформенной совместимостью.
*   **Функция**: Управляет каталогами ресурсов приложения (модели, наборы данных, кэш, временные файлы), предоставляет стандартизированную генерацию путей, управление временными рабочими областями (включая контекстный менеджер с автоматической очисткой), утилиты очистки файлов, создание/проверку контрольных сумм и отчеты об использовании диска.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, app_name: str | None = None, custom_base_dir: str | Path | None = None)`: Инициализирует пути к ресурсам.
    *   `get_model_path(self, model_name: str, create_if_missing: bool = True) -> Path`: Получает стандартизированный путь для конкретной модели.
    *   `get_dataset_path(self, dataset_name: str, create_if_missing: bool = True) -> Path`: Получает стандартизированный путь для конкретного набора данных.
    *   `get_cache_path(self, cache_name: str, create_if_missing: bool = True) -> Path`: Получает путь для файлов кэша.
    *   `create_temp_workspace(self, prefix: str = "workspace", suffix: str = "") -> Path`: Создает временный рабочий каталог.
    *   `temp_workspace(self, prefix: str = "workspace", suffix: str = "")` (контекстный менеджер): Создает временную рабочую область, которая автоматически удаляется при выходе.
    *   `clean_temp_workspace(self, workspace_path: Path, force: bool = False) -> None`: Вручную очищает временную рабочую область.
    *   `clean_old_files(self, days: int = 5, include_cache: bool = True) -> dict[str, int]`: Очищает старые загруженные и временные файлы.
    *   `get_disk_usage(self) -> dict[str, dict[str, int | str]]`: Получает статистику использования диска для всех управляемых каталогов.
    *   `create_checksum(self, file_path: str | Path, algorithm: str = "md5") -> str`: Создает контрольную сумму для файла.
    *   `verify_checksum(self, file_path: str | Path, expected_checksum: str, algorithm: str = "md5") -> bool`: Проверяет контрольную сумму файла.
    *   `get_all_directories(self) -> dict[str, Path]`: Получает все управляемые каталоги.
*   **Данные**: `app_name`, `user_data_dir`, `user_cache_dir`, `temp_dir`, `model_dir`, `dataset_dir`, `downloads_dir`.

#### 4.1.7 `core.settings.Settings`
*   **Идентификация**: `core.settings.Settings`
*   **Назначение**: Пользовательский интерфейс для всего управления конфигурацией, предоставляющий простой и стабильный способ доступа к значениям конфигурации, каталогам ресурсов и настройкам приложения.
*   **Функция**: Этот класс действует как высокоуровневый интерфейс, который делегирует сложные операции `ConfigManager` и `ResourceManager`. Он спроектирован как синглтон, доступный через функцию `get_settings`, обеспечивая единый, согласованный источник конфигурации во всем приложении.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, config_dir: str | Path | None = None)`: Инициализирует фасад Settings, настраивая базовые менеджеры.
    *   `get_config(self, path: str | None = None, default: Any = None) -> Any`: Получает значение конфигурации с использованием пути, разделенного точками.
    *   `set_config(self, path: str, value: Any) -> None`: Устанавливает значение конфигурации в памяти.
    *   `save_config(self, file_path: str | Path | None = None) -> None`: Сохраняет текущую конфигурацию в памяти в YAML-файл.
    *   `instantiate_from_config(self, config_path: str, **kwargs: Any) -> Any`: Удобный метод для создания объекта из пути его конфигурации.
    *   `get_dataset_path(self, dataset_type: str) -> Path`: Получает путь для каталога конкретного набора данных.
    *   `get_model_weights_path(self, model_type: str) -> Path`: Получает настроенный путь к файлу весов модели.
    *   `get_api_key(self, provider: str) -> str | None`: Получает API-ключ для указанного провайдера.
    *   `temp_workspace(self, prefix: str = "workspace")` (контекстный менеджер): Предоставляет временную рабочую область через `ResourceManager`.
    *   `model_dir` / `weights_dir` (свойство) `-> Path`: Каталог для весов моделей.
    *   `dataset_dir` (свойство) `-> Path`: Каталог для наборов данных.
    *   `cache_dir` (свойство) `-> Path`: Каталог кэша.
    *   `config_dir` (свойство) `-> Path`: Активный пользовательский каталог конфигурации.
    *   `species_config` (свойство) `-> SpeciesConfig`: Предоставляет отложенный доступ к помощнику `SpeciesConfig`.
*   **Интерфейсы (Используемые)**:
    *   `core.config_manager.ConfigManager`
    *   `core.resource_manager.ResourceManager`
    *   `core.species_config.SpeciesConfig`
*   **Данные**: `_instance`, `_lock`, `_initialized`.

#### 4.1.8 `core.settings.get_settings`
*   **Идентификация**: `core.settings.get_settings`
*   **Назначение**: Фабричная функция для получения экземпляра синглтона `Settings`.
*   **Функция**: Это основная точка входа для доступа ко всем настройкам приложения. Гарантирует, что класс `Settings` инициализируется только один раз.
*   **Интерфейсы (Предоставляемые)**:
    *   `get_settings(config_dir: str | Path | None = None) -> Settings`: Возвращает экземпляр синглтона `Settings`.

#### 4.1.9 `core.species_config.SpeciesConfig`
*   **Идентификация**: `core.species_config.SpeciesConfig`
*   **Назначение**: Пользовательский интерфейс для доступа к конфигурации видов.
*   **Функция**: Этот класс действует как адаптер, принимая валидированный объект Pydantic `SpeciesModel` из основной конфигурации и предоставляя простые, прямые методы для запроса данных о видах, такие как сопоставление между индексами классов и именами или получение подробных метаданных.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, config: SpeciesModel)`: Инициализируется с валидированным объектом Pydantic `SpeciesModel`.
    *   `species_map` (свойство) `-> dict[int, str]`: Получает сопоставление индексов классов с полными названиями видов.
    *   `get_species_metadata(self, species_name: str) -> dict[str, Any] | None`: Получает подробные метаданные для конкретного вида.
    *   `get_species_by_index(self, index: int) -> str | None`: Получает полное название вида по его индексу класса.
    *   `get_index_by_species(self, species_name: str) -> int | None`: Получает индекс класса для заданного названия вида.
    *   `list_species_names(self) -> list[str]`: Возвращает список всех настроенных названий видов.
*   **Интерфейсы (Используемые)**:
    *   `core.config_models.SpeciesModel`
*   **Данные**: `_config`, `_species_map`, `_reverse_species_map`, `_metadata_store`.

#### 4.1.10 `core.utils`
*   **Идентификация**: `core.utils`
*   **Назначение**: Коллекция автономных служебных функций, используемых во всей библиотеке.
*   **Функции (Предоставляемые)**:
    *   `download_file(url: str, destination: str | Path | None = None, downloads_dir: str | Path | None = None, progress_callback: Callable | None = None, chunk_size: int = 8192, timeout: int = 30, desc: str | None = None) -> Path`: Загружает файл по URL с возможностью отслеживания прогресса, чанкирования и таймаута, возвращая путь к загруженному файлу.
    *   `str_to_bgr(str_color: str) -> tuple[int, int, int]`: Преобразует строку шестнадцатеричного цвета (например, '#RRGGBB') в кортеж целых чисел BGR, подходящий для использования с такими библиотеками, как OpenCV.

#### 4.1.11 `core.weights_manager_protocol.WeightsManagerProtocol`
*   **Идентификация**: `core.weights_manager_protocol.WeightsManagerProtocol`
*   **Назначение**: Определяет протокол (интерфейс) для любого класса, который управляет весами моделей.
*   **Тип**: `typing.Protocol`
*   **Функция**: Гарантирует, что основные компоненты, такие как `BasePredictor`, могут работать с любым менеджером весов, не завися от его конкретной реализации. Это способствует слабой связанности и позволяет использовать различные стратегии управления весами (например, локальная файловая система, облачное хранилище).
*   **Интерфейсы (Предоставляемые)**:
    *   `ensure_weights(self, predictor_type: str) -> Path`: Гарантирует, что веса для данного типа модели доступны локально, потенциально загружая их, если они отсутствуют, и возвращает локальный путь к файлу весов.

### 4.2 Детальное проектирование модуля `datasets`
Модуль `datasets` отвечает за управление доступом к наборам данных и их загрузку на основе центральной конфигурации приложения. Он действует как посредник между пользовательским кодом и базовыми поставщиками данных.

**Структура проекта:**
```
datasets
├── __init__.py
└── datasets_manager.py
```

#### 4.2.1 `datasets.datasets_manager.DatasetsManager`
*   **Идентификация**: `datasets.datasets_manager.DatasetsManager`
*   **Назначение**: Управляет доступом, загрузкой и кэшированием настроенных наборов данных.
*   **Функция**: Предоставляет высокоуровневый интерфейс, который использует глобальный `Settings` для конфигурации и `ProviderService` для фактической загрузки данных. Это отделяет логику *того, какие* наборы данных доступны (определено в конфигурации) от *того, как* они загружаются и извлекаются (обрабатывается провайдерами). Он поддерживает сессионно-локальный кэш загруженных путей наборов данных, чтобы избежать повторных загрузок.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, provider_service: ProviderService)`: Инициализирует менеджер с его зависимостями.
    *   `get_dataset_info(self, dataset_name: str) -> DatasetConfig`: Извлекает модель конфигурации Pydantic для конкретного набора данных.
    *   `list_datasets(self) -> list[str]`: Перечисляет все доступные имена наборов данных из глобальной конфигурации.
    *   `list_loaded_datasets(self) -> list[str]`: Перечисляет все наборы данных, которые были загружены (загружены и кэшированы) в течение текущей сессии.
    *   `load_dataset(self, dataset_name: str, split: str | None = None, **kwargs: Any) -> Any`: Загружает конкретное разделение набора данных. Использует настроенного провайдера для загрузки набора данных, если он еще не кэширован, а затем загружает его в память. Тип возвращаемого значения зависит от реализации провайдера.
*   **Интерфейсы (Используемые)**:
    *   `core.settings.Settings`: Для доступа к конфигурациям наборов данных.
    *   `core.provider_service.ProviderService`: Для получения правильного экземпляра провайдера для загрузки и загрузки данных.
    *   `core.config_models.DatasetConfig`: Как тип возвращаемого значения для `get_dataset_info`.
*   **Данные**: `settings`, `provider_service`, `loaded_datasets` (внутренний кэш).

### 4.3 Детальное проектирование модуля `providers`
Модуль `providers` содержит конкретные реализации интерфейса `BaseProvider`. Каждый класс в этом модуле отвечает за взаимодействие с конкретным внешним источником данных (например, Hugging Face) для загрузки наборов данных или весов моделей.

**Структура проекта:**
```
providers
├── __init__.py
└── huggingface_provider.py
```

#### 4.3.1 `providers.huggingface_provider.HuggingFaceProvider`
*   **Идентификация**: `providers.huggingface_provider.HuggingFaceProvider`
*   **Наследует**: `core.base_provider.BaseProvider`
*   **Назначение**: Провайдер для загрузки и управления наборами данных и моделями HuggingFace.
*   **Функция**: Этот класс реализует интерфейс `BaseProvider` для взаимодействия с Hugging Face Hub. Он извлекает метаданные наборов данных, загружает полные наборы данных или конкретные разделения и загружает веса моделей. Он предназначен для создания `ProviderService`, который внедряет основной объект `Settings` и специфическую для провайдера конфигурацию (например, `dataset_url` и `api_key`).
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, dataset_url: str, **kwargs: Any) -> None`: Инициализирует провайдер с глобальными настройками и его специфической конфигурацией.
    *   `download_dataset(self, dataset_name: str, save_dir: str | None = None, split: str | None = None, *args: Any, **kwargs: Any) -> Path`: Загружает набор данных из Hugging Face Hub в локальный каталог.
    *   `download_model_weights(self, model_type: str, *args: Any, **kwargs: Any) -> Path`: Загружает и кэширует веса моделей из Hugging Face Hub на основе конфигурации.
    *   `get_dataset_metadata(self, dataset_name: str) -> dict[str, Any]`: Извлекает метаданные для конкретного набора данных из Hub.
    *   `get_provider_name(self) -> str`: Возвращает имя провайдера: `"huggingface"`.
    *   `load_dataset(self, dataset_path: str | Path, split: str | None = None, **kwargs) -> Any`: Загружает набор данных Hugging Face с локального дискового пути, который обычно является выходом `download_dataset`.
*   **Интерфейсы (Используемые)**:
    *   `core.base_provider.BaseProvider` (наследование).
    *   `core.settings.Settings` (для конфигурации, путей ресурсов и API-ключей).
    *   Библиотека `huggingface_hub` (для загрузки моделей и наборов данных).
    *   Библиотека `datasets` (для загрузки наборов данных с диска).
    *   Библиотека `requests` (для извлечения метаданных).
*   **Данные**: `provider_name`, `settings`, `dataset_url`, `api_key`.

### 4.4 Детальное проектирование модуля `predictors`
Модуль `predictors` содержит конкретные реализации моделей машинного обучения для различных задач анализа комаров, наследуемые от `core.base_predictor.BasePredictor`. Он также включает менеджер для обработки загрузки и локального кэширования весов моделей.

**Структура проекта:**
```
predictors
├── __init__.py
├── classifier.py
├── detector.py
├── model_weights_manager.py
└── segmenter.py
```

#### 4.4.1 `predictors.classifier.set_posix_windows` (Контекстный менеджер)
*   **Идентификация**: `predictors.classifier.set_posix_windows`
*   **Назначение**: Контекстный менеджер для обработки проблем совместимости путей при загрузке моделей FastAI, обученных на системах POSIX (Linux/macOS), в систему Windows.
*   **Функция**: Временно исправляет `pathlib.PosixPath`, чтобы он вел себя как `pathlib.WindowsPath` в системах Windows, позволяя правильно десериализовать запакованные объекты путей модели.
*   **Интерфейсы (Предоставляемые)**:
    *   `@contextmanager def set_posix_windows()`
*   **Данные**: N/A (функциональный).

#### 4.4.2 `predictors.classifier.MosquitoClassifier`
*   **Идентификация**: `predictors.classifier.MosquitoClassifier`
*   **Наследует**: `core.base_predictor.BasePredictor[ClassificationPredictionType, ClassificationGroundTruthType]`
*   **Назначение**: Классифицирует виды комаров по изображению с использованием предварительно обученной модели FastAI.
*   **Функция**: Реализует полный жизненный цикл предсказания для классификации: загрузка ученика FastAI, предсказание видов по отдельным изображениям или батчам, оценка производительности модели с подробными метриками (точность, матрица путаницы, ROC-AUC) и визуализация лучших предсказаний на изображении.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, load_model: bool = False) -> None`: Инициализирует классификатор с использованием глобальных настроек.
    *   `predict(self, input_data: np.ndarray, **kwargs: Any) -> ClassificationPredictionType`: Классифицирует одно изображение, возвращая список кортежей `(species_name, confidence)`.
    *   `predict_batch(self, input_data_batch: list[Any], show_progress: bool = False, **kwargs: Any) -> list[ClassificationPredictionType]`: Классифицирует батч изображений.
    *   `visualize(self, input_data: np.ndarray, predictions: ClassificationPredictionType, save_path: str | Path | None = None) -> np.ndarray`: Накладывает лучшие результаты классификации на изображение.
    *   `get_species_names(self) -> list[str]`: Получает отсортированный список всех известных классификатору названий видов.
    *   `get_class_index(self, species_name: str) -> int | None`: Извлекает индекс класса для заданного названия вида.
*   **Интерфейсы (Используемые)**:
    *   `core.base_predictor.BasePredictor` (наследование).
    *   `core.settings.Settings` (для конфигурации).
    *   Библиотеки FastAI и PyTorch.
    *   `numpy`, `PIL` (Pillow).
    *   `sklearn.metrics` для расширенных метрик оценки, таких как матрица путаницы и ROC-AUC.
*   **Данные**: `arch`, `data_dir`, `species_map`, `num_classes`, `learner` (объект ученика FastAI).

#### 4.4.3 `predictors.detector.MosquitoDetector`
*   **Идентификация**: `predictors.detector.MosquitoDetector`
*   **Наследует**: `core.base_predictor.BasePredictor[DetectionPredictionType, DetectionGroundTruthType]`
*   **Назначение**: Обнаруживает комаров на изображениях с использованием модели YOLO.
*   **Функция**: Загружает модель YOLO и реализует полный жизненный цикл предсказания для детекции объектов. Предсказывает ограничивающие рамки на отдельных изображениях или батчах (используя собственное батчирование YOLO), визуализирует результаты и оценивает производительность детекции (точность, полнота, AP, средний IoU).
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, load_model: bool = False) -> None`: Инициализирует детектор.
    *   `predict(self, input_data: np.ndarray, **kwargs: Any) -> DetectionPredictionType`: Обнаруживает комаров на одном изображении, возвращая список ограничивающих рамок `(cx, cy, w, h, conf)`.
    *   `predict_batch(self, input_data_batch: list[np.ndarray], show_progress: bool = True, **kwargs: Any) -> list[DetectionPredictionType]`: Обнаруживает комаров в батче изображений.
    *   `visualize(self, input_data: np.ndarray, predictions: DetectionPredictionType, save_path: str | Path | None = None) -> np.ndarray`: Рисует предсказанные ограничивающие рамки на изображении.
*   **Интерфейсы (Используемые)**:
    *   `core.base_predictor.BasePredictor` (наследование).
    *   `core.settings.Settings` (для конфигурации).
    *   Библиотека Ultralytics YOLO.
    *   `numpy`, `cv2` (OpenCV для визуализации).
*   **Данные**: `confidence_threshold`, `iou_threshold`, `max_detections`, `model` (экземпляр модели YOLO).

#### 4.4.4 `predictors.model_weights_manager.ModelWeightsManager`
*   **Идентификация**: `predictors.model_weights_manager.ModelWeightsManager`
*   **Реализует**: `core.weights_manager_protocol.WeightsManagerProtocol`
*   **Назначение**: Управляет загрузкой и локальной доступностью весов моделей.
*   **Функция**: Действует как мост между моделью, которой нужны ее веса, и `ProviderService`, который может их загрузить. Когда модель запрашивает свои веса, этот менеджер проверяет, существует ли файл уже локально. Если нет, он использует настроенного провайдера для загрузки весов и возвращает окончательный, проверенный локальный путь.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, provider_service: ProviderService)`: Инициализирует с зависимостями.
    *   `ensure_weights(self, model_type: str) -> Path`: Гарантирует, что веса модели существуют локально, загружая их при необходимости, и возвращает путь.
*   **Интерфейсы (Используемые)**:
    *   `core.settings.Settings` (чтобы получить конфигурации моделей).
    *   `core.provider_service.ProviderService` (чтобы получить экземпляр провайдера для загрузки).
    *   `core.base_provider.BaseProvider` (через сервис провайдера).
*   **Данные**: `settings`, `provider_service`.

#### 4.4.5 `predictors.segmenter.MosquitoSegmenter`
*   **Идентификация**: `predictors.segmenter.MosquitoSegmenter`
*   **Наследует**: `core.base_predictor.BasePredictor[SegmentationPredictionType, SegmentationGroundTruthType]`
*   **Назначение**: Сегментирует комаров на изображениях с использованием модели SAM2.
*   **Функция**: Загружает модель SAM2 и предоставляет методы для генерации масок сегментации. Может выполнять сегментацию всего изображения или управляться ограничивающими рамками от детектора. Также реализует визуализацию масок и оценку на основе метрик, таких как IoU.
*   **Интерфейсы (Предоставляемые)**:
    *   `__init__(self, settings: Settings, load_model: bool = False) -> None`: Инициализирует сегментатор.
    *   `predict(self, input_data: np.ndarray, **kwargs: Any) -> np.ndarray`: Генерирует бинарную маску сегментации. Может управляться `detection_boxes`, предоставленными в kwargs.
    *   `visualize(self, input_data: np.ndarray, predictions: SegmentationPredictionType, save_path: str | Path | None = None) -> np.ndarray`: Накладывает маску сегментации в виде цветного слоя на исходное изображение.
*   **Интерфейсы (Используемые)**:
    *   `core.base_predictor.BasePredictor` (наследование).
    *   `core.settings.Settings` (для конфигурации).
    *   `segment_anything` (SAM2) библиотека и PyTorch.
    *   `numpy`.
*   **Данные**: `model` (экземпляр/предсказатель модели SAM2).

## 5. Прослеживаемость

Этот раздел устанавливает всеобъемлющую прослеживаемость между функциональными требованиями системы, архитектурными целями и их соответствующими реализующими компонентами. Матрица прослеживаемости демонстрирует, как каждое проектное решение непосредственно отвечает конкретным требованиям и обеспечивает полное покрытие предполагаемой функциональности системы.

### 5.1 Сопоставление требований с компонентами

Следующая таблица сопоставляет каждое функциональное требование с его реализующими компонентами, обеспечивая четкую видимость того, как архитектура удовлетворяет целям системы:

| Функциональное требование | Основной компонент(ы) | Вспомогательные компоненты | Детали реализации |
|------------------------|---------------------|----------------------|----------------------|
| **Централизованное управление конфигурацией** | `core.ConfigManager` | `core.settings.Settings`, модели Pydantic в `core.config_models` | ConfigManager предоставляет синглтон-доступ к конфигурации, в то время как Settings обрабатывает параметры, специфичные для среды, а модели Pydantic обеспечивают типобезопасность и валидацию |
| **Гибкий доступ и управление наборами данных** | `datasets.DatasetsManager` | модуль `providers`, `core.BaseProvider`, `core.ProviderService` | DatasetsManager управляет операциями с наборами данных, BaseProvider определяет интерфейс провайдера, конкретные провайдеры обрабатывают определенные источники данных |
| **Стандартизированный интерфейс моделей** | `core.BasePredictor` | Все реализации моделей | Абстрактный базовый класс обеспечивает согласованный API для всех моделей предсказания со стандартизированными методами для predict, visualize и evaluate |
| **Классификация видов** | `predictors.MosquitoClassifier` | `core.BasePredictor`, `core.ModelWeightsManager` | Реализует логику классификации, специфичную для видов, с поддержкой нескольких архитектур моделей и управления весами |
| **Детекция объектов** | `predictors.MosquitoDetector` | `core.BasePredictor`, утилиты визуализации | Предоставляет детекцию ограничивающих рамок с возможностью пакетной обработки и интегрированной визуализации |
| **Сегментация изображений** | `predictors.MosquitoSegmenter` | `core.BasePredictor`, утилиты постобработки | Реализует сегментацию на уровне пикселей с поддержкой нескольких форматов вывода и обработки масок |
| **Эффективная пакетная обработка** | методы `predict_batch()` | `core.BasePredictor`, управление ресурсами | Оптимизированная пакетная обработка с управлением памятью и отслеживанием прогресса для всех типов моделей |
| **Визуализация результатов** | методы `visualize()` | Библиотеки построения графиков, `core.BasePredictor` | Согласованный интерфейс визуализации с настраиваемыми форматами вывода и наложением аннотаций |
| **Оценка производительности модели** | `evaluate()` и `evaluate_batch()` | Расчет метрик, утилиты валидации | Всеобъемлющая структура оценки со стандартными метриками и пользовательскими протоколами оценки |
| **Управление ресурсами** | `core.ResourceManager` | Утилиты файловой системы, управление путями | Кросс-платформенная обработка ресурсов с автоматическим созданием и очисткой каталогов |
| **Управление весами моделей** | `predictors.ModelWeightsManager` | `core.WeightsManagerProtocol`, утилиты загрузки | Автоматизированная загрузка весов моделей, кэширование и управление версиями с проверкой целостности |
| **Интеграция с внешними источниками данных** | `core.BaseProvider` | `providers.HuggingFaceProvider`, `core.ProviderService` | Расширяемая система провайдеров, поддерживающая несколько источников данных с едиными шаблонами доступа |
| **Обработка метаданных видов** | `core.SpeciesConfig` | Система конфигурации, валидация | Управление структурированной информацией о видах с таксономической валидацией и обогащением метаданных |
| **Кросс-платформенная совместимость** | Все основные модули | Платформо-специфичные утилиты | Согласованное поведение в разных операционных системах с учетом платформы при обработке файлов |

### 5.2 Прослеживаемость архитектурных целей

Архитектура решает несколько неявных, но критически важных целей, которые обеспечивают долгосрочную жизнеспособность и удобство использования системы:

#### 5.2.1 Модульность и расширяемость
**Цель**: Обеспечить легкое добавление новых моделей, источников данных и функциональности без изменения существующего кода.

**Реализация**:
- **Базовые абстракции**: `BasePredictor`, `BaseProvider` и классы протоколов определяют четкие контракты
- **Разделение модулей**: Отдельные модули `core`, `datasets`, `providers` и `predictors` с минимальными взаимозависимостями
- **Архитектура плагинов**: Система провайдеров позволяет добавлять новые источники данных через простую реализацию интерфейса
- **Иерархия наследования**: Хорошо определенные базовые классы позволяют создавать новые типы моделей с минимальным шаблонным кодом

**Верификация**: Новые модели для видов комаров могут быть добавлены путем наследования от `BasePredictor` без изменения существующих компонентов.

#### 5.2.2 Поведение на основе конфигурации
**Цель**: Максимизировать гибкость системы через внешнюю конфигурацию вместо жестко заданных значений.

**Реализация**:
- **Централизованные настройки**: `core.ConfigManager` предоставляет единый источник истины для конфигурации
- **Осведомленность о среде**: `core.Settings` адаптируется к различным средам развертывания
- **Типобезопасность**: Модели Pydantic в `core.config_models` обеспечивают валидацию конфигурации
- **Иерархическая конфигурация**: Поддержка глобальных, модульно-специфичных и экземплярно-специфичных настроек

**Верификация**: Пользователи могут изменять поведение системы через файлы конфигурации без изменений кода.

#### 5.2.3 Надежная обработка ошибок и управление ресурсами
**Цель**: Обеспечить стабильность системы и правильную очистку ресурсов при любых условиях.

**Реализация**:
- **Управление ресурсами**: `core.ResourceManager` обрабатывает операции файловой системы с надлежащей очисткой
- **Управление весами**: `ModelWeightsManager` управляет файлами моделей с проверкой целостности
- **Устойчивость провайдеров**: Сбои источников данных обрабатываются корректно с механизмами отката
- **Управление памятью**: Пакетная обработка включает оптимизацию памяти и процедуры очистки

**Верификация**: Система продолжает работать и правильно очищает ресурсы даже при сбое отдельных компонентов.

#### 5.2.4 Оптимизация производительности
**Цель**: Обеспечить возможности эффективной обработки как для единичных предсказаний, так и для пакетных операций.

**Реализация**:
- **Пакетная обработка**: Все модели реализуют оптимизированные методы пакетной обработки с управлением памятью
- **Ленивая загрузка**: Модели и наборы данных загружаются только при необходимости для минимизации использования памяти
- **Стратегия кэширования**: Часто запрашиваемые данные кэшируются надлежащим образом для сокращения избыточных операций
- **Пул ресурсов**: Общие ресурсы эффективно управляются между несколькими операциями

**Верификация**: Пакетные операции эффективно масштабируются с размером набора данных при сохранении разумного использования памяти.

#### 5.2.5 Опыт разработчика и удобство использования
**Цель**: Предоставить интуитивно понятные API и всеобъемлющую функциональность для исследователей и разработчиков.

**Реализация**:
- **Согласованный интерфейс**: Все модели разделяют общие сигнатуры методов (predict, visualize, evaluate)
- **Богатая визуализация**: Встроенные возможности визуализации с настраиваемыми форматами вывода
- **Всесторонняя оценка**: Стандартные метрики и пользовательские протоколы оценки
- **Интеграция документации**: Структура кода поддерживает всестороннюю генерацию документации

**Верификация**: Новые пользователи могут выполнять общие задачи с минимальным обучением и четкими шаблонами API.

### 5.3 Прослеживаемость сквозных аспектов

Несколько общесистемных проблем решаются за счет согласованной реализации в нескольких компонентах:

#### 5.3.1 Целостность потока данных
- **Источник**: Модуль `providers` обеспечивает подлинность данных
- **Обработка**: `predictors` сохраняет происхождение данных через конвейер обработки
- **Хранение**: `ResourceManager` обрабатывает сохранение данных с проверкой целостности
- **Валидация**: Система конфигурации проверяет форматы данных и параметры

#### 5.3.2 Соображения масштабируемости
- **Горизонтальное масштабирование**: Система провайдеров поддерживает распределенные источники данных
- **Вертикальное масштабирование**: Пакетная обработка оптимизирует использование памяти и вычислений
- **Стратегия кэширования**: Многоуровневое кэширование сокращает избыточные операции
- **Управление ресурсами**: Эффективная очистка предотвращает утечки ресурсов

#### 5.3.3 Поддерживаемость и тестирование
- **Четкое разделение**: Модульный дизайн позволяет целенаправленное тестирование отдельных компонентов
- **Внедрение зависимостей**: Конфигурационно-управляемые зависимости поддерживают изоляцию тестов
- **Соответствие протоколу**: Абстрактные базовые классы определяют тестируемые контракты
- **Поддержка моков**: Абстракция провайдеров позволяет всестороннее модульное тестирование

### 5.4 Верификация соответствия

Каждое требование может быть проверено через конкретные артефакты реализации:

1. **Соответствие интерфейсу**: Абстрактные базовые классы определяют контракты, которые должны выполнять конкретные реализации
2. **Покрытие конфигурации**: Все настраиваемое поведение раскрывается через систему настроек
3. **Обработка ошибок**: Каждый компонент включает соответствующую обработку исключений и очистку ресурсов
4. **Метрики производительности**: Методы пакетной обработки включают мониторинг производительности и оптимизацию
5. **Точки расширения**: Новая функциональность может быть добавлена через четко определенные механизмы расширения

Эта матрица прослеживаемости гарантирует, что каждый аспект проектирования системы непосредственно отвечает выявленным требованиям, сохраняя при этом архитектурную целостность и поддерживая будущие улучшения.
