"""Pydantic models for configuration validation.

This module defines all Pydantic models used to parse and validate the
YAML configuration files. `CulicidaeLabConfig` serves as the root model
that encompasses all others.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TaxonomyModel(BaseModel):
    """Defines the taxonomic classification of a species."""

    family: str
    subfamily: str
    genus: str
    subgenus: str | None = None
    species_complex: str | None = None


class SpeciesAttributesModel(BaseModel):
    """Defines attributes found in the nested 'metadata' key for each species."""

    vector_status: bool
    diseases: list[str]
    habitat: str
    breeding_sites: list[str]
    sources: list[str]


class SingleSpeciesMetadataModel(BaseModel):
    """Represents the full metadata object for a single species."""

    common_name: str
    taxonomy: TaxonomyModel
    metadata: SpeciesAttributesModel


class SpeciesFiles(BaseModel):
    """A helper model representing the contents of YAML files in the species dir."""

    model_config = ConfigDict(extra="allow")
    species_info_mapping: dict[str, str] = {}
    species_metadata: dict[str, SingleSpeciesMetadataModel] = {}


class SpeciesModel(BaseModel):
    """Configuration for the entire 'species' section."""

    model_config = ConfigDict(extra="allow")
    species_classes: dict[int, str] = Field(default_factory=dict)
    species_metadata: SpeciesFiles = Field(default_factory=SpeciesFiles)


class AppSettings(BaseSettings):
    """Core application settings, loaded from env vars or a config file."""

    model_config = SettingsConfigDict(env_prefix="CULICIDAELAB_", extra="ignore")
    environment: str = "production"
    log_level: str = "INFO"


class ProcessingConfig(BaseModel):
    """General processing parameters."""

    batch_size: int = 32
    confidence_threshold: float = 0.5
    device: str = "cpu"


class VisualizationConfig(BaseModel):
    """Configuration for visualization settings."""

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


class PredictorConfig(BaseModel):
    """Configuration for a single predictor."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    target_: str = Field(..., alias="_target_")
    model_path: str
    confidence: float = 0.5
    device: str = "cpu"
    params: dict[str, Any] = {}
    repository_id: str | None = None
    filename: str | None = None
    provider_name: str | None = None
    model_arch: str | None = None
    model_config_path: str | None
    model_config_filename: str | None
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    model_config = ConfigDict(extra="allow")
    name: str
    path: str
    format: str
    classes: list[str]
    provider_name: str


class ProviderConfig(BaseModel):
    """Configuration for a data provider."""

    model_config = ConfigDict(extra="allow")
    target_: str = Field(..., alias="_target_")
    dataset_url: str
    api_key: str | None = None


class CulicidaeLabConfig(BaseModel):
    """The root Pydantic model for all CulicidaeLab configurations.

    It validates the entire configuration structure after loading from YAML files.
    """

    model_config = ConfigDict(extra="allow")
    app_settings: AppSettings = Field(default_factory=AppSettings)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    datasets: dict[str, DatasetConfig] = {}
    predictors: dict[str, PredictorConfig] = {}
    providers: dict[str, ProviderConfig] = {}
    species: SpeciesModel = Field(default_factory=SpeciesModel)
