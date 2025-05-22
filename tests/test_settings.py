"""
Tests for the settings module.
"""

import json
import pytest
import yaml  # type: ignore

from culicidaelab.species_cofig import SpeciesConfig


@pytest.fixture
def temp_config_yaml(tmp_path):
    """Create a temporary YAML config file."""
    config = {
        "species_map": {
            0: "Aedes Aegypti",
            1: "Anopheles Gambiae",
            2: "Culex Quinquefasciatus",
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    return config_path


@pytest.fixture
def temp_config_json(tmp_path):
    """Create a temporary JSON config file."""
    config = {
        "species_map": {
            "0": "Aedes Aegypti",
            "1": "Anopheles Gambiae",
            "2": "Culex Quinquefasciatus",
        },
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory with species subdirectories."""
    species = ["aedes_aegypti", "anopheles_gambiae", "culex_quinquefasciatus"]
    for species_name in species:
        species_dir = tmp_path / species_name
        species_dir.mkdir()
        # Create dummy image files
        (species_dir / "image1.jpg").touch()
        (species_dir / "image2.jpg").touch()
    return tmp_path


def test_init_empty():
    """Test initialization without config or data directory."""
    config = SpeciesConfig()
    assert config.get_species_map() == {}
    assert config.get_num_species() == 0
    assert config.get_species_list() == []


def test_load_from_yaml(temp_config_yaml):
    """Test loading configuration from YAML file."""
    config = SpeciesConfig(config_path=temp_config_yaml)
    assert config.get_num_species() == 3
    assert config.get_species_name(0) == "Aedes Aegypti"
    assert config.get_species_name(1) == "Anopheles Gambiae"
    assert config.get_species_name(2) == "Culex Quinquefasciatus"


def test_load_from_json(temp_config_json):
    """Test loading configuration from JSON file."""
    config = SpeciesConfig(config_path=temp_config_json)
    assert config.get_num_species() == 3
    assert config.get_species_name(0) == "Aedes Aegypti"
    assert config.get_species_name(1) == "Anopheles Gambiae"
    assert config.get_species_name(2) == "Culex Quinquefasciatus"


def test_infer_from_directory(temp_data_dir):
    """Test inferring species from directory structure."""
    config = SpeciesConfig(data_dir=temp_data_dir)
    assert config.get_num_species() == 3
    species_list = config.get_species_list()
    assert "Aedes Aegypti" in species_list
    assert "Anopheles Gambiae" in species_list
    assert "Culex Quinquefasciatus" in species_list


def test_save_to_file(tmp_path):
    """Test saving configuration to file."""
    config = SpeciesConfig()
    config.add_species("Aedes Aegypti")
    config.add_species("Anopheles Gambiae")

    # Test YAML
    yaml_path = tmp_path / "test_config.yaml"
    config.save_to_file(yaml_path)
    assert yaml_path.exists()

    # Test JSON
    json_path = tmp_path / "test_config.json"
    config.save_to_file(json_path)
    assert json_path.exists()


def test_add_remove_species():
    """Test adding and removing species."""
    config = SpeciesConfig()

    # Test adding species
    config.add_species("Aedes Aegypti")
    assert config.get_num_species() == 1
    assert config.get_species_name(0) == "Aedes Aegypti"

    # Test adding another species
    config.add_species("Anopheles Gambiae")
    assert config.get_num_species() == 2

    # Test removing species
    config.remove_species("Aedes Aegypti")
    assert config.get_num_species() == 1
    assert "Aedes Aegypti" not in config.get_species_list()


def test_species_index_lookup():
    """Test species index and name lookup."""
    config = SpeciesConfig()
    config.add_species("Aedes Aegypti")

    # Test getting index
    assert config.get_species_index("Aedes Aegypti") == 0
    assert config.get_species_index("Nonexistent Species") is None

    # Test getting name
    assert config.get_species_name(0) == "Aedes Aegypti"
    assert config.get_species_name(999) is None


def test_invalid_config_file(tmp_path):
    """Test handling of invalid configuration files."""
    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        SpeciesConfig(config_path="nonexistent.yaml")

    # Test invalid file extension
    invalid_path = tmp_path / "config.txt"
    invalid_path.touch()
    with pytest.raises(ValueError):
        SpeciesConfig(config_path=invalid_path)

    # Test invalid YAML content
    invalid_yaml = tmp_path / "invalid.yaml"
    with open(invalid_yaml, "w") as f:
        f.write("invalid: yaml: content:")
    with pytest.raises(yaml.YAMLError):
        SpeciesConfig(config_path=invalid_yaml)


def test_invalid_data_directory():
    """Test handling of invalid data directory."""
    with pytest.raises(FileNotFoundError):
        SpeciesConfig(data_dir="nonexistent_directory")
