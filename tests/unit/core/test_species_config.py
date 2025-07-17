import pytest

from culicidaelab.core.config_models import SpeciesModel
from culicidaelab.core.species_config import SpeciesConfig


@pytest.fixture
def mock_species_model():
    """Provides a validated SpeciesModel instance for testing."""
    data = {
        "species_classes": {
            0: "aedes_aegypti",
            1: "aedes_albopictus",
            2: "anopheles_gambiae",
        },
        "species_metadata": {
            "species_info_mapping": {
                "aedes_aegypti": "Aedes aegypti",
                "aedes_albopictus": "Aedes albopictus",
            },
            "species_metadata": {
                "Aedes aegypti": {
                    "common_name": "Yellow Fever Mosquito",
                    "taxonomy": {"family": "Culicidae", "subfamily": "Culicinae", "genus": "Aedes"},
                    "metadata": {
                        "vector_status": True,
                        "diseases": ["dengue", "zika"],
                        "habitat": "urban",
                        "breeding_sites": ["tires"],
                        "sources": [],
                    },
                },
                "Aedes albopictus": {
                    "common_name": "Asian Tiger Mosquito",
                    "taxonomy": {"family": "Culicidae", "subfamily": "Culicinae", "genus": "Aedes"},
                    "metadata": {
                        "vector_status": True,
                        "diseases": ["dengue", "chikungunya"],
                        "habitat": "suburban",
                        "breeding_sites": ["containers"],
                        "sources": [],
                    },
                },
            },
        },
    }
    return SpeciesModel.model_validate(data)


@pytest.fixture
def species_config(mock_species_model):
    """Provides a SpeciesConfig instance initialized with the mock model."""
    return SpeciesConfig(mock_species_model)


def test_species_map_creation(species_config):
    expected_map = {
        0: "Aedes aegypti",
        1: "Aedes albopictus",
        2: "anopheles_gambiae",
    }
    assert species_config.species_map == expected_map


def test_get_species_metadata(species_config):
    metadata = species_config.get_species_metadata("Aedes aegypti")
    assert metadata is not None
    assert metadata["common_name"] == "Yellow Fever Mosquito"
    assert metadata["metadata"]["vector_status"] is True

    assert species_config.get_species_metadata("Culex quinquefasciatus") is None


def test_get_species_by_index(species_config):
    assert species_config.get_species_by_index(0) == "Aedes aegypti"
    assert species_config.get_species_by_index(1) == "Aedes albopictus"
    assert species_config.get_species_by_index(2) == "anopheles_gambiae"
    assert species_config.get_species_by_index(99) is None


def test_get_index_by_species(species_config):
    assert species_config.get_index_by_species("Aedes aegypti") == 0
    assert species_config.get_index_by_species("Aedes albopictus") == 1
    assert species_config.get_index_by_species("anopheles_gambiae") == 2
    assert species_config.get_index_by_species("Culex pipiens") is None


def test_list_species_names(species_config):
    expected_names = ["Aedes aegypti", "Aedes albopictus", "anopheles_gambiae"]
    assert sorted(species_config.list_species_names()) == sorted(expected_names)
