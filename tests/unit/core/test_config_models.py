import pytest
from pydantic import ValidationError

from culicidaelab.core.config_models import (
    PredictorConfig,
    DatasetConfig,
    ProviderConfig,
    SpeciesModel,
    CulicidaeLabConfig,
    SingleSpeciesMetadataModel,
    SpeciesFiles,
)


def test_predictor_config_validation():
    valid_data = {
        "target": "culicidaelab.predictors.Classifier",
        "model_path": "/path/to/model.pt",
        "confidence": 0.6,
        "device": "cuda",
        "repository_id": "org/repo",
        "filename": "model.pt",
        "model_config_path": "dummy/path/config.yaml",
        "model_config_filename": "config.yaml",
    }
    predictor = PredictorConfig(**valid_data)
    assert predictor.target == valid_data["target"]
    assert predictor.confidence == 0.6

    with pytest.raises(ValidationError):
        PredictorConfig(model_path="/path/to/model.pt")


def test_dataset_config_validation():
    valid_data = {
        "name": "classification",
        "repository": "default_repo",
        "path": "culicidae-classification-5-class",
        "format": "huggingface",
        "classes": ["aedes aegypti", "aedes albopictus"],
        "provider_name": "huggingface",
    }
    dataset = DatasetConfig(**valid_data)
    assert dataset.name == "classification"
    assert len(dataset.classes) == 2

    with pytest.raises(ValidationError):
        DatasetConfig(
            path="path",
            format="huggingface",
            classes=[],
            provider_name="huggingface",
        )


def test_provider_config_validation():
    valid_data = {
        "target": "culicidaelab.providers.HuggingFaceProvider",
        "dataset_url": "https://api.huggingface.co/datasets/{dataset_name}",
        "api_key": "some_key",
    }
    provider = ProviderConfig(**valid_data)
    assert provider.target == valid_data["target"]
    assert provider.api_key == "some_key"

    valid_data_no_key = {
        "target": "culicidaelab.providers.HuggingFaceProvider",
        "dataset_url": "https://api.huggingface.co/datasets/{dataset_name}",
    }
    provider_no_key = ProviderConfig(**valid_data_no_key)
    assert provider_no_key.api_key is None


def test_species_config_models():
    single_species_data = {
        "common_name": "Yellow Fever Mosquito",
        "taxonomy": {
            "family": "Culicidae",
            "subfamily": "Culicinae",
            "genus": "Aedes",
            "subgenus": "Stegomyia",
        },
        "metadata": {
            "vector_status": True,
            "diseases": ["dengue", "zika"],
            "habitat": "urban",
            "breeding_sites": ["tires"],
            "sources": ["source1"],
        },
    }
    species_meta = SingleSpeciesMetadataModel(**single_species_data)
    assert species_meta.taxonomy.genus == "Aedes"
    assert "dengue" in species_meta.metadata.diseases

    species_files_data = {
        "species_info_mapping": {"aedes_aegypti": "Aedes aegypti"},
        "species_metadata": {"Aedes aegypti": single_species_data},
    }
    species_files = SpeciesFiles(**species_files_data)
    assert species_files.species_info_mapping["aedes_aegypti"] == "Aedes aegypti"

    species_model_data = {
        "species_classes": {0: "aedes_aegypti"},
        "species_metadata": species_files_data,
    }
    species_model = SpeciesModel(**species_model_data)
    assert species_model.species_classes[0] == "aedes_aegypti"
    assert species_model.species_metadata.species_metadata["Aedes aegypti"].common_name == "Yellow Fever Mosquito"


def test_root_config_model():
    full_config_data = {
        "app_settings": {"environment": "testing"},
        "processing": {"batch_size": 16},
        "datasets": {
            "classification": {
                "name": "classification",
                "path": "path/to/data",
                "format": "hf",
                "classes": ["a", "b"],
                "provider_name": "huggingface",
                "repository": "default-repo",  # Added repository field
            },
        },
        "predictors": {
            "classifier": {
                "target": "some.Classifier",
                "model_path": "path/to/model.pt",
                "model_config_path": "dummy/path/config.yaml",
                "model_config_filename": "config.yaml",
            },
        },
        "providers": {
            "huggingface": {
                "target": "some.Provider",
                "dataset_url": "http://example.com",
            },
        },
        "species": {
            "species_classes": {0: "aedes_aegypti"},
            "species_metadata": {
                "species_info_mapping": {"aedes_aegypti": "Aedes aegypti"},
                "species_metadata": {},
            },
        },
    }
    root_config = CulicidaeLabConfig(**full_config_data)
    assert root_config.app_settings.environment == "testing"
    assert root_config.processing.batch_size == 16
    assert "classification" in root_config.datasets
    assert "classifier" in root_config.predictors
    assert "huggingface" in root_config.providers
    assert root_config.species.species_classes[0] == "aedes_aegypti"
