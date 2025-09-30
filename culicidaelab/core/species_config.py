"""Species configuration management and access module.

This module provides a facade pattern implementation for managing and accessing species
configuration data in the Culicidae Lab system. It simplifies the interaction with
species-related data by providing a clean interface for accessing species information,
metadata, and class mappings.

The module abstracts the complexity of the underlying configuration model and provides
user-friendly methods for common operations like looking up species by index,
retrieving metadata, and managing species name mappings.

Example:
    ```python
    from culicidaelab.core.config_models import SpeciesModel
    from culicidaelab.core.species_config import SpeciesConfig

    # Create a species config instance
    species_config = SpeciesConfig(species_model)

    # Get species name by index
    species_name = species_config.get_species_by_index(0)  # Returns e.g., "Aedes aegypti"

    # Get metadata for a species
    metadata = species_config.get_species_metadata("Aedes aegypti")
    ```
"""

from typing import Any

from culicidaelab.core.config_models import SingleSpeciesMetadataModel, SpeciesModel


class SpeciesConfig:
    """A user-friendly facade for accessing and managing species configuration data.

    This class implements the Facade pattern to simplify access to species-related
    configuration data. It provides an intuitive interface for managing species
    information, including class mappings, metadata, and name translations.

    Args:
        config (SpeciesModel): A validated Pydantic model containing the complete
            species configuration data.

    Attributes:
        _config (SpeciesModel): The source configuration model containing raw data.
        _species_map (dict[int, str]): Maps numeric class indices to full species names.
        _reverse_species_map (dict[str, int]): Maps full species names to their numeric indices.
        _metadata_store (dict): Contains detailed metadata for each species.
        class_to_full_name_map (dict[str, str]): Maps short class names to full scientific names.
        reverse_class_to_full_name_map (dict[str, str]): Maps full scientific names to short class names.

    Example:
        ```python
        config = SpeciesModel(...)  # Your validated config
        species_helper = SpeciesConfig(config)

        # Get full name for a class index
        species_name = species_helper.get_species_by_index(0)

        # Get metadata for a species
        metadata = species_helper.get_species_metadata(species_name)
        ```
    """

    def __init__(self, config: SpeciesModel):
        """Initializes the species configuration helper.

        Sets up internal mappings and data structures for efficient species data access.
        Processes the input configuration to create bidirectional mappings between
        species names, class names, and indices.

        Args:
            config (SpeciesModel): The validated species configuration model.
        """
        self._config = config
        self._species_map: dict[int, str] = {}
        self.class_to_full_name_map = self._config.species_metadata.species_info_mapping
        self.reverse_class_to_full_name_map = {v: k for k, v in self.class_to_full_name_map.items()}

        for idx, class_name in self._config.species_classes.items():
            full_name = self.class_to_full_name_map.get(class_name, class_name)
            self._species_map[idx] = full_name

        self._reverse_species_map: dict[str, int] = {name: idx for idx, name in self._species_map.items()}
        self._metadata_store: dict[
            str,
            SingleSpeciesMetadataModel,
        ] = self._config.species_metadata.species_metadata

    @property
    def species_map(self) -> dict[int, str]:
        """Gets the mapping of class indices to full, human-readable species names.

        Returns:
            dict[int, str]: A dictionary mapping numeric class indices to full
                scientific species names.

        Example:
            ```python
            species_config = SpeciesConfig(config)
            mapping = species_config.species_map
            # Returns: {0: "Aedes aegypti", 1: "Aedes albopictus"}
            ```
        """
        return self._species_map

    def get_index_by_species(self, species_name: str) -> int | None:
        """Gets the numeric class index for a given species name.

        Looks up the numeric class index used by the model for a given full
        species name. This is useful for mapping between model predictions
        and species names.

        Args:
            species_name (str): The full scientific name of the species
                (e.g., "Aedes aegypti").

        Returns:
            int | None: The numeric class index used by the model, or None if the
                species is not found in the configuration.

        Example:
            ```python
            index = species_config.get_index_by_species("Aedes aegypti")
            # Returns: 0
            ```
        """
        return self._reverse_species_map.get(species_name)

    def get_species_by_index(self, index: int) -> str | None:
        """Gets the full scientific species name for a given class index.

        Converts a numeric class index used by the model into the corresponding
        full scientific species name. This is particularly useful when processing
        model predictions.

        Args:
            index (int): The numeric class index used by the model.

        Returns:
            str | None: The full scientific species name as a string, or None if the
                index is not found in the configuration.

        Example:
            ```python
            species = species_config.get_species_by_index(0)
            # Returns: "Aedes aegypti"
            ```
        """
        return self._species_map.get(index)

    def get_species_label(self, species_name: str) -> str:
        """Gets the short label/class name for a given full species name.

        Converts a full scientific species name to its corresponding short label
        used in the dataset and model classifications.

        Args:
            species_name (str): The full scientific name of the species
                (e.g., "Aedes aegypti").

        Returns:
            str: The short label/class name used in the dataset
                (e.g., "ae_aegypti").

        Example:
            ```python
            label = species_config.get_species_label("Aedes aegypti")
            # Returns: "ae_aegypti"
            ```
        """
        return self.reverse_class_to_full_name_map[species_name]

    def get_species_metadata(self, species_name: str) -> dict[str, Any] | None:
        """Gets the detailed metadata for a specific species.

        Retrieves comprehensive metadata about a species, including taxonomic
        information, characteristics, and any custom metadata fields defined
        in the configuration.

        Args:
            species_name (str): The full scientific name of the species
                (e.g., "Aedes aegypti").

        Returns:
            dict[str, Any] | None: A dictionary containing all metadata fields for the
                species, or None if the species is not found. The dictionary structure
                depends on the metadata fields defined in the configuration.

        Example:
            ```python
            metadata = species_config.get_species_metadata("Aedes aegypti")
            # Returns: {
            #     "family": "Culicidae",
            #     "genus": "Aedes",
            #     "species": "aegypti",
            #     "common_name": "Yellow fever mosquito",
            #     ...
            # }
            ```
        """
        model_object = self._metadata_store.get(species_name)
        return model_object.model_dump() if model_object else None

    def list_species_names(self) -> list[str]:
        """Returns a list of all configured full species names.

        Provides a complete list of all species names that are configured in the system.
        The names are returned in their full scientific format.

        Returns:
            list[str]: A list of full scientific species names configured in the system.

        Example:
            ```python
            species_list = species_config.list_species_names()
            # Returns: ["Aedes aegypti", "Aedes albopictus", ...]
            ```
        """
        return list(self._reverse_species_map.keys())
