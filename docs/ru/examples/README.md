# Примеры использования

Добро пожаловать в раздел с примерами использования, где мы переходим от теории к практике. Здесь вы найдете серию практических руководств, разработанных для ознакомления вас с основными функциями `CulicidaeLab`. Каждый пример является самодостаточным, исполняемым и основывается на предыдущем, демонстрируя полный и логичный рабочий процесс для анализа изображений комаров.

Мы будем следовать типичному реальному сценарию, взяв изображение и обработав его от начала до конца. Вы узнаете, как:

1.  **Использование модуля `settings`:** Мы начнем с основ. Вы научитесь получать доступ к конфигурациям и понимать, как библиотека управляет ресурсами. Это центральная нервная система `CulicidaeLab`.
2.  **Управлять наборами данных**: Это руководство покажет вам, как использовать `DatasetsManager` для поиска, загрузки и кэширования наборов данных, настроенных в библиотеке. Это необходимо для всех, кто хочет проводить крупномасштабную оценку моделей или исследовательский анализ данных (EDA).
3.  **Обнаруживать комаров:** Далее мы воспользуемся `MosquitoDetector`, чтобы ответить на первый важный вопрос: «Есть ли на этом изображении комар и где он находится?»
4.  **Сегментировать комаров:** для наиболее детального анализа мы будем использовать `MosquitoSegmenter` для создания точной, на уровне пикселей, маски точной формы комара.
5.  **Классифицировать виды комаров:** Как только комар будет найден, мы воспользуемся `MosquitoClassifier` для определения его вида, что является решающим шагом для эпидемиологических исследований.

Прежде чем погрузиться в примеры, пожалуйста, убедитесь, что вы успешно установили `CulicidaeLab`. Если вы этого еще не сделали, пожалуйста, обратитесь к **Руководству по установке**.

В примерах используются образцы изображений, которые, как предполагается, находятся в локальном каталоге `test_imgs/`. Вы можете скачать их из нашего репозитория на GitHub или просто заменить пути к файлам на свои собственные изображения.

Мы рекомендуем вам самостоятельно запустить фрагменты кода в Jupyter Notebook или Python-скрипте, чтобы получить представление о том, как все работает вместе. Давайте начнем
