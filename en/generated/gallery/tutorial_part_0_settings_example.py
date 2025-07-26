"""
# Understanding and Using the Settings

This tutorial demonstrates how to use the core `settings` object in CulicidaeLab.
The `settings` object is the main entry point for accessing configurations, file paths,
and model parameters throughout the library.
"""
# %%
# Install the `culicidaelab` library if not already installed
# !pip install -q culicidaelab

# %%
import yaml
from pathlib import Path

# %%
from culicidaelab import get_settings

# %% [markdown]
# ## 1. Using Default Settings
#
# The easiest way to start with `CulicidaeLab` is by loading the default settings.
# The `get_settings()` function acts as a singleton; it loads the configuration once
# and returns the same instance on subsequent calls. This ensures a consistent
# state across your application.
#
# The default settings are loaded from the configuration files bundled with the library.

# %%
# Get the default settings instance
settings = get_settings()

# The settings object provides easy access to key resource directories.
# The library will automatically create these directories if they don't exist.
print("--- Default Resource Directories ---")
print(f"Active Config Directory: {settings.config_dir}")
print(f"Models Directory: {settings.model_dir}")
print(f"Datasets Directory: {settings.dataset_dir}")
print(f"Cache Directory: {settings.cache_dir}")

# %% [markdown]
# ## 2. Accessing Model Weight Paths
#
# The `settings` object knows the default local paths for all predictor model weights.
# When you instantiate a predictor, it uses these paths to find or download the models.

# %%
# Get the configured local file paths for different model types
detection_weights = settings.get_model_weights_path("detector")
segmentation_weights = settings.get_model_weights_path("segmenter")
classification_weights = settings.get_model_weights_path("classifier")

print("--- Default Model Weight Paths ---")
print(f"Detection Model: {detection_weights}")
print(f"Segmentation Model: {segmentation_weights}")
print(f"Classification Model: {classification_weights}")

# %% [markdown]
# ## 3. Working with Species Configuration
#
# All species-related information, including class names and detailed metadata,
# is managed through the `species_config` property. This is crucial for interpreting
# the output of the classification model.

# %%
# Get the dedicated species configuration object
species_config = settings.species_config

# You can easily retrieve the mapping of class indices to species names.
print("\n--- Species Index-to-Name Mapping ---")
for idx, species in species_config.species_map.items():
    print(f"Class {idx}: {species}")

# You can also fetch detailed metadata for any specific species.
species_name = "Aedes aegypti"
metadata = species_config.get_species_metadata(species_name)
if isinstance(metadata, dict):
    print(f"\n--- Metadata for '{species_name}' ---")
    for key, value in metadata.items():
        print(f"{key}: {value}")

# %% [markdown]
# ## 4. Using a Custom Configuration Directory
#
# For advanced use cases, such as providing your own species metadata or changing
# default model parameters, you can point the library to a custom configuration directory.
#
# `CulicidaeLab` will load your custom `.yaml` files and merge them on top of the defaults.
# This allows you to override only the settings you need to change.

# %%
# Create a custom config directory and a new config file
custom_config_dir = Path("custom_configs")
custom_config_dir.mkdir(exist_ok=True)

# Define a minimal custom configuration. We'll just override the species info.
# Any settings not defined here will fall back to the library's defaults.
example_config = {
    "species": {
        "species_classes": {0: "Aedes aegypti", 1: "Anopheles gambiae"},
        "species_metadata": {
            "species_info_mapping": {
                "aedes_aegypti": "Aedes aegypti",
                "anopheles_gambiae": "Anopheles gambiae",
            },
            "species_metadata": {
                "Aedes aegypti": {
                    "common_name": "Custom Yellow Fever Mosquito",
                    "taxonomy": {
                        "family": "Culicidae",
                        "subfamily": "Culicinae",
                        "genus": "Aedes",
                    },
                    "metadata": {
                        "vector_status": True,
                        "diseases": ["Dengue", "Zika"],
                        "habitat": "Urban",
                        "breeding_sites": ["Artificial containers"],
                        "sources": ["custom_source"],
                    },
                },
                "Anopheles gambiae": {
                    "common_name": "Custom African Malaria Mosquito",
                    "taxonomy": {
                        "family": "Culicidae",
                        "subfamily": "Anophelinae",
                        "genus": "Anopheles",
                    },
                    "metadata": {
                        "vector_status": True,
                        "diseases": ["Malaria"],
                        "habitat": "Rural",
                        "breeding_sites": ["Puddles"],
                        "sources": ["custom_source"],
                    },
                },
            },
        },
    },
}


# Write the custom config file
config_file_path = custom_config_dir / "species.yaml"
with open(config_file_path, "w") as f:
    yaml.safe_dump(example_config, f)

# Now, initialize settings with the path to our custom directory.
# `get_settings` is smart enough to create a *new* instance if a different config_dir is provided.
print("\n--- Initializing with Custom Settings ---")
custom_settings = get_settings(config_dir=str(custom_config_dir))

print(f"Active Config Directory: {custom_settings.config_dir}")

# Let's check if our custom species map was loaded
print("\n--- Custom Species Mapping ---")
for idx, species in custom_settings.species_config.species_map.items():
    print(f"Class {idx}: {species}")

# %% [markdown]
# ## 5. Overriding a Single Configuration Value
#
# Sometimes, you may only want to change a single value at runtime without creating new YAML files.
# The `set_config` method is perfect for this.
#
# Let's load the default settings and change the confidence threshold for the detector.

# %%
# Load default settings again (or use the previous 'settings' instance)
runtime_settings = get_settings()
original_confidence = runtime_settings.get_config("predictors.detector.confidence")
print(f"Original detector confidence: {original_confidence}")

# Set a new confidence value at runtime
runtime_settings.set_config("predictors.detector.confidence", 0.85)
new_confidence = runtime_settings.get_config("predictors.detector.confidence")
print(f"New detector confidence: {new_confidence}")
