from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Optional


class SpeciesConfig:
    """
    Configuration class for species data.

    Handles species mapping and metadata for mosquito species classification.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the species configuration.

        Args:
            config: Configuration containing species data
        """
        self._config = config
        self._species_map = self._load_species_map()
        self._species_metadata = self._load_species_metadata()

    def _load_species_map(self) -> Dict[int, str]:
        """
        Load species mapping from configuration.

        Returns:
            Dictionary mapping class indices to species names
        """
        if not hasattr(self._config, "species") or not hasattr(self._config.species, "classes"):
            return {}

        species_map = {}
        for idx, species in enumerate(self._config.species.classes):
            species_map[idx] = species

        return species_map

    def _load_species_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load species metadata from configuration.

        Returns:
            Dictionary mapping species names to their metadata
        """
        if not hasattr(self._config, "species") or not hasattr(self._config.species, "metadata"):
            return {}

        return OmegaConf.to_container(self._config.species.metadata, resolve=True) or {}

    @property
    def species_map(self) -> Dict[int, str]:
        """
        Get the mapping of class indices to species names.

        Returns:
            Dictionary mapping class indices to species names
        """
        return self._species_map

    def get_species_metadata(self, species_name: str) -> Dict[str, Any]:
        """
        Get metadata for a specific species.

        Args:
            species_name: Name of the species

        Returns:
            Dictionary containing metadata for the species
        """
        if species_name not in self._species_metadata:
            return {}

        return self._species_metadata[species_name]

    def get_species_by_index(self, index: int) -> Optional[str]:
        """
        Get species name by class index.

        Args:
            index: Class index

        Returns:
            Species name or None if index not found
        """
        return self._species_map.get(index)

    def get_index_by_species(self, species_name: str) -> Optional[int]:
        """
        Get class index by species name.

        Args:
            species_name: Name of the species

        Returns:
            Class index or None if species not found
        """
        for idx, name in self._species_map.items():
            if name == species_name:
                return idx
        return None
