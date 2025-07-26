"""Facade for accessing species configuration.

This module provides the `SpeciesConfig` class, which simplifies access to
species information, such as metadata and class mappings, by abstracting it
from the main configuration model.
"""

from typing import Any

from culicidaelab.core.config_models import SingleSpeciesMetadataModel, SpeciesModel


class SpeciesConfig:
    """A user-friendly facade for accessing species configuration.

    This class acts as an adapter, taking the complex, validated `SpeciesModel`
    object and providing simple methods and properties for accessing species data.

    Args:
        config (SpeciesModel): A validated `SpeciesModel` Pydantic object from
            the main settings.

    Attributes:
        _config (SpeciesModel): The source configuration model.
        _species_map (dict[int, str]): A mapping of class indices to full species names.
        _reverse_species_map (dict[str, int]): A reverse mapping of species
            names to indices.
        _metadata_store (dict): A store for species metadata.
    """

    def __init__(self, config: SpeciesModel):
        """Initializes the species configuration helper."""
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

        Example:
            {0: "Aedes aegypti", 1: "Aedes albopictus"}
        """
        return self._species_map

    def get_index_by_species(self, species_name: str) -> int | None:
        """Gets the class index by its full species name.

        Args:
            species_name (str): The full name of the species.

        Returns:
            int | None: The integer class index, or None if not found.
        """
        return self._reverse_species_map.get(species_name)

    def get_species_by_index(self, index: int) -> str | None:
        """Gets the full species name by its class index.

        Args:
            index (int): The integer class index.

        Returns:
            str | None: The full species name as a string, or None if not found.
        """
        return self._species_map.get(index)

    def get_species_label(self, species_name: str) -> str:
        """Gets label for the full species name (e.g., "Aedes aegypti").
        Args:
            species_name (str): The full name of the species (e.g., "Aedes aegypti").
        Returns:
            str: The dataset label for the species.
        """
        return self.reverse_class_to_full_name_map[species_name]

    def get_species_metadata(self, species_name: str) -> dict[str, Any] | None:
        """Gets the detailed metadata for a specific species as a dictionary.

        Args:
            species_name (str): The full name of the species (e.g., "Aedes aegypti").

        Returns:
            dict[str, Any] | None: A dictionary representing the species metadata,
            or None if not found.
        """
        model_object = self._metadata_store.get(species_name)
        return model_object.model_dump() if model_object else None

    def list_species_names(self) -> list[str]:
        """Returns a list of all configured full species names.

        Returns:
            list[str]: A list of strings, where each string is a species name.
        """
        return list(self._reverse_species_map.keys())
