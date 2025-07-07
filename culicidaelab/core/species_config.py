from typing import Any

from .config_models import SpeciesModel, SingleSpeciesMetadataModel


class SpeciesConfig:
    """
    A user-friendly facade for accessing species configuration.

    This class acts as an adapter, taking the complex, validated SpeciesModel
    object and providing simple, convenient methods and properties for accessing
    species maps and metadata.
    """

    def __init__(self, config: SpeciesModel):
        """
        Initialize the species configuration helper.

        This constructor performs the necessary data transformations to create
        user-friendly mappings from the raw Pydantic models.

        Args:
            config: A validated SpeciesModel Pydantic object from the main settings.
        """
        self._config = config

        # --- Create the user-friendly species_map ---
        # This combines the class index map with the full name map.
        self._species_map: dict[int, str] = {}
        class_to_full_name_map = self._config.species_metadata.species_info_mapping

        for idx, class_name in self._config.species_classes.items():
            # Look up the full name, falling back to the class name if not found
            full_name = class_to_full_name_map.get(class_name, class_name)
            self._species_map[idx] = full_name

        # --- Create a reverse map for convenience ---
        self._reverse_species_map: dict[str, int] = {name: idx for idx, name in self._species_map.items()}

        # --- Store a direct reference to the detailed metadata dictionary ---
        self._metadata_store: dict[str, SingleSpeciesMetadataModel] = self._config.species_metadata.species_metadata

    @property
    def species_map(self) -> dict[int, str]:
        """
        Get the mapping of class indices to full, human-readable species names.
        Example: {0: "Aedes aegypti", 1: "Aedes albopictus"}
        """
        return self._species_map

    def get_species_metadata(self, species_name: str) -> dict[str, Any] | None:
        """
        Get the detailed metadata for a specific species as a dictionary.

        Args:
            species_name: The full name of the species (e.g., "Aedes aegypti").

        Returns:
            A dictionary representing the species metadata, or None if not found.
        """
        model_object = self._metadata_store.get(species_name)
        if model_object:
            # Use model_dump() to convert the Pydantic model to a dict
            return model_object.model_dump()
        return None

    def get_species_by_index(self, index: int) -> str | None:
        """Gets the full species name by its class index.

        Args:
            index: The integer class index.

        Returns:
            The full species name as a string, or None if not found.
        """
        return self._species_map.get(index)

    def get_index_by_species(self, species_name: str) -> int | None:
        """Gets the class index by its full species name.

        Args:
            species_name: The full name of the species.

        Returns:
            The integer class index, or None if not found.
        """
        return self._reverse_species_map.get(species_name)

    def list_species_names(self) -> list[str]:
        """Returns a list of all configured full species names.

        Returns:
            A list of strings, where each string is a species name.
        """
        return list(self._reverse_species_map.keys())
