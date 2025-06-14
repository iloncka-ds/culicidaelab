from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any, List, Optional


class TaxonomyModel(BaseModel):
    """Defines the taxonomic classification of a species."""

    family: str
    subfamily: str
    genus: str
    subgenus: Optional[str] = None
    species_complex: Optional[str] = None


class SpeciesAttributesModel(BaseModel):
    """
    Defines the attributes found inside the nested 'metadata' key
    for each species.
    """

    vector_status: bool
    diseases: List[str]
    habitat: str
    breeding_sites: List[str]
    sources: List[str]


class SingleSpeciesMetadataModel(BaseModel):
    """
    Represents the full metadata object for a single species.
    This model corresponds to the value for a key like "Aedes aegypti".
    """

    common_name: str
    taxonomy: TaxonomyModel
    metadata: SpeciesAttributesModel

class SpeciesFiles(BaseModel):
    """
    A helper model that represents the contents of a single YAML file in the species directory.
    This is necessary because the `species_metadata.yaml` file contains two top-level keys.
    """

    model_config = ConfigDict(extra="allow")

    # This field will capture the `species_info_mapping` block.
    species_info_mapping: Dict[str, str] = {}

    # This field will capture the `species_metadata` block.
    species_metadata: Dict[str, SingleSpeciesMetadataModel] = {}


class SpeciesModel(BaseModel):
    """
    Configuration for the entire 'species' section.
    This model now correctly reflects that its direct children are named
    after the YAML files that were loaded.
    """

    model_config = ConfigDict(extra="allow")

    # This field captures the content of 'species_classes.yaml'
    # It expects a dictionary mapping integers to strings.
    species_classes: Dict[int, str] = Field(default_factory=dict)

    # This field captures the content of 'species_metadata.yaml'
    # Its value is validated by the SpeciesFiles helper model.
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
    device: str = "auto"


class PredictorConfig(BaseModel):
    """Configuration for a single predictor."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    target_: str = Field(..., alias="_target_")
    model_path: str
    confidence: float = 0.5
    device: str = "auto"
    params: Dict[str, Any] = {}
    repository_id: Optional[str] = None
    filename: Optional[str] = None


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    model_config = ConfigDict(extra="allow")

    name: str
    path: str
    format: str
    classes: List[str]


class ProviderConfig(BaseModel):
    """Configuration for a data provider."""

    model_config = ConfigDict(extra="allow")

    target_: str = Field(..., alias="_target_")
    api_key: Optional[str] = None


class CulicidaeLabConfig(BaseModel):
    """
    The root Pydantic model for all CulicidaeLab configurations.
    It validates the entire configuration structure after loading from YAML files.
    """

    model_config = ConfigDict(extra="allow")

    app_settings: AppSettings = Field(default_factory=AppSettings)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    datasets: Dict[str, DatasetConfig] = {}
    predictors: Dict[str, PredictorConfig] = {}
    providers: Dict[str, ProviderConfig] = {}
    species: SpeciesModel = Field(default_factory=SpeciesModel)
