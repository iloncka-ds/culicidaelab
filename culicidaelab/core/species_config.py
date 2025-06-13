from typing import Dict, Any, Optional

from .config_models import SpeciesModel


class SpeciesConfig:
    """
    Handles species mapping and metadata logic.
    This class is now a stateless helper that operates on a validated Pydantic model.
    """

    def __init__(self, config: SpeciesModel):
        """
        Initialize the species configuration helper.

        Args:
            config: A validated SpeciesModel Pydantic object.
        """
        self._config = config
        self._species_map = {idx: name for idx, name in enumerate(self._config.classes)}
        self._reverse_species_map = {name: idx for idx, name in self._species_map.items()}

    @property
    def species_map(self) -> Dict[int, str]:
        """Get the mapping of class indices to species names."""
        return self._species_map

    def get_species_metadata(self, species_name: str) -> Dict[str, Any]:
        """Get metadata for a specific species."""
        return self._config.metadata.get(species_name, {})

    def get_species_by_index(self, index: int) -> Optional[str]:
        """Get species name by class index."""
        return self._species_map.get(index)

    def get_index_by_species(self, species_name: str) -> Optional[int]:
        """Get class index by species name."""
        return self._reverse_species_map.get(species_name)
