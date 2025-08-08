"""Pydantic models for configuration validation.

This module defines all Pydantic models used to parse and validate the
YAML configuration files. `CulicidaeLabConfig` serves as the root model
that encompasses all others, including a versioning field to ensure
future compatibility.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# --- Versioning ---
# A simple string to identify the configuration schema version.
# When a breaking change is made to any model, this version should be incremented.
CONFIG_SCHEMA_VERSION = "1.0"


# --- Species Metadata Models ---


class TaxonomyModel(BaseModel):
    """Defines the detailed taxonomic classification of a species.

    Args:
        family (str): The taxonomic family (e.g., "Culicidae").
        subfamily (str): The taxonomic subfamily (e.g., "Culicinae").
        genus (str): The taxonomic genus (e.g., "Aedes").
        subgenus (str | None): The optional subgenus. Defaults to None.
        species_complex (str | None): The optional species complex. Defaults to None.
    """

    family: str
    subfamily: str
    genus: str
    subgenus: str | None = None
    species_complex: str | None = None


class SpeciesAttributesModel(BaseModel):
    """Defines biological and ecological attributes for a species.

    Args:
        vector_status (bool): True if the species is a known disease vector.
        diseases (list[str]): A list of diseases the species is known to transmit.
        habitat (str): A description of the species' typical habitat.
        breeding_sites (list[str]): Common breeding sites for the species.
        sources (list[str]): A list of URLs or citations for the data sources.
    """

    vector_status: bool
    diseases: list[str]
    habitat: str
    breeding_sites: list[str]
    sources: list[str]


class SingleSpeciesMetadataModel(BaseModel):
    """Represents the full metadata object for a single species.

    Args:
        common_name (str): The common name for the species.
        taxonomy (TaxonomyModel): The detailed taxonomic hierarchy.
        metadata (SpeciesAttributesModel): Additional biological attributes.
    """

    common_name: str
    taxonomy: TaxonomyModel
    metadata: SpeciesAttributesModel


class SpeciesFiles(BaseModel):
    """A helper model representing the aggregated contents of species YAML files.

    Args:
        species_info_mapping (dict[str, str]): A mapping from species class name to
            its corresponding metadata file.
        species_metadata (dict[str, SingleSpeciesMetadataModel]): A dictionary holding
            the fully parsed metadata for each species.
    """

    model_config = ConfigDict(extra="allow")
    species_info_mapping: dict[str, str] = {}
    species_metadata: dict[str, SingleSpeciesMetadataModel] = {}


class SpeciesModel(BaseModel):
    """Configuration for the entire 'species' section of the config.

    Args:
        species_classes (dict[int, str]): A mapping of integer class IDs to
            string-based species names.
        species_metadata (SpeciesFiles): The aggregated species metadata loaded
            from the species directory.
    """

    model_config = ConfigDict(extra="allow")
    species_classes: dict[int, str] = Field(default_factory=dict)
    species_metadata: SpeciesFiles = Field(default_factory=SpeciesFiles)


# --- Core and Tooling Models ---


class AppSettings(BaseSettings):
    """Core application settings, primarily loaded from environment variables.

    These settings control the runtime behavior of the library, such as logging.
    They can be set via environment variables prefixed with `CULICIDAELAB_`.

    Args:
        environment (str): The runtime environment (e.g., "development", "production").
        log_level (str): The logging level (e.g., "INFO", "DEBUG").
    """

    model_config = SettingsConfigDict(env_prefix="CULICIDAELAB_", extra="ignore")
    environment: str = "production"
    log_level: str = "INFO"


class ProcessingConfig(BaseModel):
    """General data processing parameters.

    Args:
        batch_size (int): The number of samples to process in a single batch.
        confidence_threshold (float): The minimum confidence score for predictions.
        device (str): The compute device to use ("cpu" or "cuda").
    """

    batch_size: int = 32
    confidence_threshold: float = 0.5
    device: str = "cpu"


class VisualizationConfig(BaseModel):
    """Configuration for generating visual outputs like overlays and plots.

    Args:
        overlay_color (str): The default hex color for segmentation masks.
        alpha (float): The opacity level for overlays (0.0 to 1.0).
        box_color (str): The hex color for drawing bounding boxes.
        text_color (str): The hex color for label text.
        font_scale (float): The font size scaling factor.
        box_thickness (int): The line thickness for bounding boxes.
        text_thickness (int | None): The line thickness for text. Defaults to 2.
        format (str | None): The output image format (e.g., "png", "jpg"). Defaults to "png".
        dpi (int | None): The resolution in dots per inch for saved figures. Defaults to 300.
    """

    model_config = ConfigDict(extra="allow")
    overlay_color: str = "#000000"
    alpha: float = 0.5
    box_color: str = "#000000"
    text_color: str = "#000000"
    font_scale: float = 0.5
    box_thickness: int = 2
    text_thickness: int | None = 2
    format: str | None = "png"
    dpi: int | None = 300


# --- Main Component Models ---


class PredictorConfig(BaseModel):
    """Configuration for a single inference predictor.

    This model defines how to load and use a specific pre-trained model for inference.

    Args:
        target_ (str): The fully qualified import path to the predictor class
            (e.g., `culicidaelab.models.YOLOv8Predictor`). Aliased from `_target_`.
        model_path (str): The local path or remote URL to the model weights file.
        confidence (float): The default confidence threshold for this predictor.
        device (str): The compute device to use ("cpu" or "cuda").
        params (dict[str, Any]): A dictionary of extra parameters to pass to the
            predictor's constructor.
        repository_id (str | None): The Hugging Face Hub repository ID for the model.
        filename (str | None): The specific model filename within the repository.
        provider_name (str | None): The name of the provider (e.g., "huggingface").
        model_arch (str | None): The model architecture name (e.g., "yolov8n-seg").
        model_config_path (str | None): The path to the model's specific config file.
        model_config_filename (str | None): The filename of the model's config.
        visualization (VisualizationConfig): Custom visualization settings for this predictor.
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    target_: str = Field(..., alias="_target_")
    model_path: str
    confidence: float = 0.5
    device: str = "cpu"
    params: dict[str, Any] = Field(default_factory=dict)
    repository_id: str | None = None
    filename: str | None = None
    provider_name: str | None = None
    model_arch: str | None = None
    model_config_path: str | None = None
    model_config_filename: str | None = None
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)


class DatasetConfig(BaseModel):
    """Configuration for a single dataset.

    Args:
        name (str): The unique internal name for the dataset.
        path (str): The local directory path for storing the dataset.
        format (str): The dataset format (e.g., "imagefolder", "coco", "yolo").
        classes (list[str]): A list of class names present in the dataset.
        provider_name (str): The name of the data provider (e.g., "huggingface").
        repository (str): The repository ID on the provider's platform.
        derived_datasets (list[str] | None): A list of Hugging Face repository IDs
            for datasets that were derived from this one. Defaults to None.
        trained_models_repositories (list[str] | None): A list of Hugging Face
            repository IDs for models trained on this dataset. Defaults to None.
    """

    model_config = ConfigDict(extra="allow")
    name: str
    path: str
    format: str
    classes: list[str]
    provider_name: str
    repository: str
    derived_datasets: list[str] | None = None
    trained_models_repositories: list[str] | None = None


class ProviderConfig(BaseModel):
    """Configuration for a data provider, such as Hugging Face.

    Args:
        target_ (str): The fully qualified import path to the provider's
            service class. Aliased from `_target_`.
        dataset_url (str): The base URL for accessing datasets from this provider.
        api_key (str | None): An optional API key for authentication, if required.
    """

    model_config = ConfigDict(extra="allow")
    target_: str = Field(..., alias="_target_")
    dataset_url: str
    api_key: str | None = None


# --- Root Configuration Model ---


class CulicidaeLabConfig(BaseModel):
    """The root Pydantic model for all CulicidaeLab configurations.

    This model validates the entire configuration structure after it is loaded
    from YAML files, serving as the single source of truth for all settings.

    Args:
        config_version (str): The version of the configuration schema. This is used
            to ensure compatibility with the library version.
        app_settings (AppSettings): Core application settings.
        processing (ProcessingConfig): Default processing parameters.
        datasets (dict[str, DatasetConfig]): A mapping of dataset names to their configs.
        predictors (dict[str, PredictorConfig]): A mapping of predictor names to their configs.
        providers (dict[str, ProviderConfig]): A mapping of provider names to their configs.
        species (SpeciesModel): Configuration and metadata related to all species.
    """

    model_config = ConfigDict(extra="allow")
    config_version: str = Field(default=CONFIG_SCHEMA_VERSION)
    app_settings: AppSettings = Field(default_factory=AppSettings)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    datasets: dict[str, DatasetConfig] = Field(default_factory=dict)
    predictors: dict[str, PredictorConfig] = Field(default_factory=dict)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    species: SpeciesModel = Field(default_factory=SpeciesModel)
