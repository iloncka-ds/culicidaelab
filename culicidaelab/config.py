"""
Configuration module for CulicidaeLab.
"""

from __future__ import annotations

import json
from pathlib import Path
import yaml  # type: ignore


class SpeciesConfig:
    """Class to manage species configuration and mapping."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ):
        """
        Initialize species configuration.

        Args:
            config_path (str, optional): Path to configuration file (YAML or JSON)
            data_dir (str, optional): Path to data directory to infer species from structure
        """
        self.species_map: dict[int, str] = {}

        if config_path:
            # Convert to Path and ensure it's a Path object
            config_path = Path(config_path)
            self.load_from_file(config_path)
        elif data_dir:
            # Convert to Path and ensure it's a Path object
            data_dir = Path(data_dir)
            self.infer_from_directory(data_dir)

    def load_from_file(self, config_path: str | Path) -> dict[int, str]:
        """
        Load species mapping from a configuration file.

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            Dict mapping class indices to species names
        """
        config_path = Path(config_path)  # Convert to Path object
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load based on file extension
        if config_path.suffix.lower() in [".yml", ".yaml"]:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            with open(config_path) as f:
                config = json.load(f)
        else:
            raise ValueError("Configuration file must be YAML or JSON")

        # Extract species mapping
        if "species_map" in config:
            # Convert string keys to integers if needed
            self.species_map = {int(k) if isinstance(k, str) else k: v for k, v in config["species_map"].items()}
        else:
            raise KeyError("Configuration file must contain 'species_map' key")

        return self.species_map

    def infer_from_directory(self, data_dir: str | Path) -> dict[int, str]:
        """
        Infer species mapping from directory structure.
        Expects a directory structure like:
        data_dir/
            species_name1/
                image1.jpg
                image2.jpg
            species_name2/
                image3.jpg
                ...

        Args:
            data_dir: Path to data directory

        Returns:
            Dict mapping class indices to species names
        """
        data_dir = Path(data_dir)  # Convert to Path object
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Get all subdirectories (species folders)
        species_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

        # Sort for consistent indexing
        species_dirs.sort()

        # Create mapping
        self.species_map = {idx: dir_name.name.replace("_", " ").title() for idx, dir_name in enumerate(species_dirs)}

        return self.species_map

    def save_to_file(self, config_path: str | Path):
        """
        Save current species mapping to a configuration file.

        Args:
            config_path: Path to save configuration file
        """
        config_path = Path(config_path)  # Convert to Path object

        # Create config dictionary
        config = {
            "species_map": self.species_map,
            "metadata": {
                "num_species": len(self.species_map),
                "species_list": list(self.species_map.values()),
            },
        }

        # Save based on file extension
        if config_path.suffix.lower() in [".yml", ".yaml"]:
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
        elif config_path.suffix.lower() == ".json":
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
        else:
            raise ValueError("Configuration file must be YAML or JSON")

    def get_species_map(self) -> dict[int, str]:
        """Get the current species mapping."""
        return self.species_map

    def get_num_species(self) -> int:
        """Get the number of species in the mapping."""
        return len(self.species_map)

    def get_species_list(self) -> list:
        """Get list of all species names."""
        return list(self.species_map.values())

    def add_species(self, species_name: str):
        """
        Add a new species to the mapping.

        Args:
            species_name: Name of the species to add
        """
        if species_name not in self.species_map.values():
            new_idx = max(self.species_map.keys(), default=-1) + 1
            self.species_map[new_idx] = species_name

    def remove_species(self, species_name: str):
        """
        Remove a species from the mapping.

        Args:
            species_name: Name of the species to remove
        """
        keys_to_remove = [k for k, v in self.species_map.items() if v == species_name]
        for key in keys_to_remove:
            del self.species_map[key]

    def get_species_index(self, species_name: str) -> int | None:
        """
        Get the index for a given species name.

        Args:
            species_name: Name of the species

        Returns:
            Index of the species if found, None otherwise
        """
        for idx, name in self.species_map.items():
            if name == species_name:
                return idx
        return None

    def get_species_name(self, index: int) -> str | None:
        """
        Get the species name for a given index.

        Args:
            index: Index of the species

        Returns:
            Name of the species if found, None otherwise
        """
        return self.species_map.get(index)
