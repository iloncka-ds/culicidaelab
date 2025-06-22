# Species Configuration Module

## Overview
The `species_config` module provides a structured way to manage species-related configurations in the culicidaelab library. It handles the mapping between species names and their corresponding class indices, as well as storing and retrieving species metadata.

## Features
- **Species Mapping**: Bidirectional mapping between class indices and species names
- **Metadata Management**: Store and retrieve species-specific metadata
- **Type Safety**: Type hints and validation for configuration data
- **OmegaConf Integration**: Seamless integration with OmegaConf configuration system

## Installation
```bash
pip install culicidaelab
```

## Quick Start

### Basic Usage
```python
from omegaconf import OmegaConf
from culicidaelab.core.species_config import SpeciesConfig

# Example configuration
config = OmegaConf.create({
    "species": {
        "classes": ["aedes_aegypti", "anopheles_gambiae", "culex_quinquefasciatus"],
        "metadata": {
            "aedes_aegypti": {
                "common_name": "Yellow fever mosquito",
                "disease_vectors": ["dengue", "zika", "chikungunya"]
            },
            "anopheles_gambiae": {
                "common_name": "African malaria mosquito",
                "disease_vectors": ["malaria"]
            },
            "culex_quinquefasciatus": {
                "common_name": "Southern house mosquito",
                "disease_vectors": ["West Nile virus", "lymphatic filariasis"]
            }
        }
    }
})

# Initialize species configuration
species_config = SpeciesConfig(config)

# Get species by index
species = species_config.get_species_by_index(0)  # Returns "aedes_aegypti"

# Get index by species
index = species_config.get_index_by_species("anopheles_gambiae")  # Returns 1

# Get species metadata
metadata = species_config.get_species_metadata("culex_quinquefasciatus")
# Returns: {"common_name": "Southern house mosquito", "disease_vectors": [...]}
```

## API Reference

### `SpeciesConfig` Class

#### Constructor
```python
SpeciesConfig(config: DictConfig)
```

**Parameters**:
- `config`: OmegaConf DictConfig containing species configuration

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `species_map` | `Dict[int, str]` | Read-only property that returns the mapping of class indices to species names |

#### Methods

##### `get_species_by_index(index: int) -> Optional[str]`
Get the species name for a given class index.

**Parameters**:
- `index`: Integer class index

**Returns**:
- Species name as string, or None if index not found

##### `get_index_by_species(species_name: str) -> Optional[int]`
Get the class index for a given species name.

**Parameters**:
- `species_name`: Name of the species

**Returns**:
- Integer class index, or None if species not found

##### `get_species_metadata(species_name: str) -> Dict[str, Any]`
Get metadata for a specific species.

**Parameters**:
- `species_name`: Name of the species

**Returns**:
- Dictionary containing metadata for the species, or empty dict if not found

## Configuration Structure

The `SpeciesConfig` class expects a configuration object with the following structure:

```yaml
species:
  classes:  # List of species names
    - "aedes_aegypti"
    - "anopheles_gambiae"
    - "culex_quinquefasciatus"

  metadata:  # Optional metadata for each species
    aedes_aegypti:
      common_name: "Yellow fever mosquito"
      disease_vectors:
        - "dengue"
        - "zika"
        - "chikungunya"
    anopheles_gambiae:
      common_name: "African malaria mosquito"
      disease_vectors:
        - "malaria"
    culex_quinquefasciatus:
      common_name: "Southern house mosquito"
      disease_vectors:
        - "West Nile virus"
        - "lymphatic filariasis"
```

## Advanced Usage

### Custom Metadata
You can extend the metadata for each species with any additional fields:

```yaml
species:
  classes: ["aedes_aegypti"]
  metadata:
    aedes_aegypti:
      common_name: "Yellow fever mosquito"
      scientific_name: "Aedes aegypti"
      distribution: ["tropical", "subtropical"]
      breeding_sites: ["artificial containers", "stagnant water"]
      active_hours: "daytime"
      lifespan_days: 30
      wing_length_mm: 3.0
      disease_vectors:
        - name: "dengue"
          transmission_risk: "high"
        - name: "zika"
          transmission_risk: "moderate"
```

### Integration with ConfigManager
```python
from culicidaelab.core import ConfigManager
from culicidaelab.core.species_config import SpeciesConfig

# Initialize config manager
config_manager = ConfigManager()
config = config_manager.load_config()

# Create species config
species_config = SpeciesConfig(config)

# Use in your application
def classify_mosquito(image):
    # Get model prediction (pseudo-code)
    prediction = model.predict(image)
    species_idx = prediction.argmax()
    species_name = species_config.get_species_by_index(int(species_idx))

    if species_name:
        metadata = species_config.get_species_metadata(species_name)
        return {
            "species": species_name,
            "common_name": metadata.get("common_name", "Unknown"),
            "diseases": metadata.get("disease_vectors", [])
        }
    return {"error": "Species not recognized"}
```

## Best Practices

1. **Consistent Naming**:
   - Use consistent naming conventions for species (e.g., snake_case)
   - Prefer scientific names over common names when possible

2. **Metadata Organization**:
   - Group related metadata fields together
   - Use consistent data types for the same fields across species
   - Document the expected metadata structure

3. **Error Handling**:
   - Always check if a species exists before accessing its metadata
   - Provide meaningful error messages for missing or invalid data

## Performance Considerations

1. **Initialization**:
   - The species map is built once during initialization
   - Metadata is loaded lazily when first accessed

2. **Lookup Operations**:
   - Index to species name lookups are O(1)
   - Species name to index lookups are O(n) in the number of species
   - For large numbers of species, consider using a reverse lookup dictionary

## Example: Adding New Species

1. Update the configuration file:
```yaml
species:
  classes:
    - "aedes_aegypti"
    - "anopheles_gambiae"
    - "culex_quinquefasciatus"
    - "aedes_albopictus"  # New species

  metadata:
    aedes_albopictus:
      common_name: "Asian tiger mosquito"
      disease_vectors:
        - "dengue"
        - "chikungunya"
        - "zika"
      distribution: ["tropical", "temperate"]
```

2. The new species will be automatically available in the application:
```python
# Get the index of the new species
index = species_config.get_index_by_species("aedes_albopictus")  # Returns 3

# Get metadata for the new species
metadata = species_config.get_species_metadata("aedes_albopictus")
```

## Troubleshooting

### Common Issues

1. **Species Not Found**
   - Verify the species name matches exactly (case-sensitive)
   - Check for typos in the configuration file
   - Ensure the species is listed in the `classes` array

2. **Missing Metadata**
   - Check if metadata exists for the species
   - Verify the metadata structure matches what your code expects

3. **Configuration Loading**
   - Ensure the configuration is properly loaded before creating a SpeciesConfig instance
   - Check for YAML syntax errors in the configuration file

## Contributing

When modifying the species configuration:
1. Update the documentation to reflect any changes
2. Add or update unit tests
3. Maintain backward compatibility when possible
4. Document any breaking changes

---
*Documentation generated on: 2025-05-28*
