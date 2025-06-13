from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any, List


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

    _target_: str = Field(..., alias="_target_")
    model_path: str
    confidence: float = 0.5
    device: str = "auto"
    params: Dict[str, Any] = {}


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""

    name: str
    path: str
    format: str
    classes: List[str]


class ProviderConfig(BaseModel):
    """Configuration for a data provider."""

    _target_: str = Field(..., alias="_target_")
    # API key can be loaded from env via AppSettings
    api_key: str | None = None


class SpeciesModel(BaseModel):
    """Configuration for species classes and metadata."""

    classes: List[str] = []
    metadata: Dict[str, Dict[str, Any]] = {}


# --- Top Level Configuration Model ---
class CulicidaeLabConfig(BaseModel):
    """
    The root Pydantic model for all CulicidaeLab configurations.
    It validates the entire configuration structure after loading from YAML files.
    """
    model_config = SettingsConfigDict(env_prefix="CULICIDAELAB_", extra="allow")

    app_settings: AppSettings = Field(default_factory=AppSettings)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    datasets: Dict[str, DatasetConfig] = {}
    predictors: Dict[str, PredictorConfig] = {}
    providers: Dict[str, ProviderConfig] = {}
    species: SpeciesModel = Field(default_factory=SpeciesModel)


