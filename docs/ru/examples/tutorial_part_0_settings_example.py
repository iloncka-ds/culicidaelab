"""
# CulicidaeLab Settings Module Example

This notebook demonstrates how to use the settings module in CulicidaeLab.
"""

import os
import yaml

from culicidaelab.core.settings import get_settings

# ## 1. Using Default Settings
#
# Get the default settings instance. This will use configurations from the default_configs directory.

# +
# Get default settings
settings = get_settings()

# Print some basic information
print(f"Config directory: {settings.config_dir}")
print(f"Models directory: {settings.model_dir}")
print(f"Datasets directory: {settings.dataset_dir}")
print(f"Cache directory: {settings.cache_dir}")
# -

# ## 2. Working with Model Weights
#
# Demonstrate how to get model weights paths and handle downloads.

# +
# Get paths for different model types
detection_weights = settings.get_model_weights_path("detector")
segmentation_weights = settings.get_model_weights_path("segmenter")
classification_weights = settings.get_model_weights_path("classifier")

print("Model weights paths:")
print(f"Detection: {detection_weights}")
print(f"Segmentation: {segmentation_weights}")
print(f"Classification: {classification_weights}")
# -

# ## 3. Working with Species Configuration
#
# Access species information from the configuration.

# +
# Get species configuration
species_config = settings.species_config

# Print species mapping
print("Species mapping:")
for idx, species in species_config.species_map.items():
    print(f"Class {idx}: {species}")

# Print metadata for a specific species
species_name = "Aedes aegypti"
metadata = species_config.get_species_metadata(species_name)
if isinstance(metadata, dict):
    print(f"\nMetadata for {species_name}:")
    for key, value in metadata.items():
        print(f"{key}: {value}")
# -

# ## 4. Using Custom Configuration
#
# Demonstrate how to use custom configuration directory.

# +

# Create a custom config directory
custom_config_dir = "custom_configs"
os.makedirs(custom_config_dir, exist_ok=True)

# Example minimal config with required "species" key
example_config = {
    "species": {
        "species_map": {0: "Aedes aegypti", 1: "Anopheles gambiae"},
        "metadata": {
            "Aedes aegypti": {"color": "yellow", "region": "tropical"},
            "Anopheles gambiae": {"color": "brown", "region": "Africa"},
        },
    },
}

# Write example config.yaml if it doesn't exist
config_file_path = os.path.join(custom_config_dir, "config.yaml")
if not os.path.exists(config_file_path):
    with open(config_file_path, "w") as f:
        yaml.safe_dump(example_config, f)

# Validate the custom configuration directory contains the required key
required_key = "species"
with open(config_file_path) as file:
    config_data = yaml.safe_load(file)
    if required_key not in config_data:
        raise KeyError(
            f"Missing required key '{required_key}' in custom configuration file.",
        )

# Get settings with custom config directory
custom_settings = get_settings(config_dir=custom_config_dir)

print(f"Custom config directory: {custom_settings.config_dir}")

# Note: This will use default configs if custom configs are not found
print(f"Using default configs: {custom_settings.config_dir == settings.config_dir}")
