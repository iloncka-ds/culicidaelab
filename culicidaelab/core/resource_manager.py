"""
Centralized resource management for models, datasets, and temporary files.

This module provides cross-platform resource management with proper error handling,
logging, and comprehensive path management capabilities.
"""

import os
import shutil
import platform
import tempfile
import time
import logging
from pathlib import Path
from contextlib import contextmanager
import appdirs
import toml
from threading import Lock
import hashlib

logger = logging.getLogger(__name__)


class ResourceManagerError(Exception):
    """Custom exception for ResourceManager operations."""

    pass


class ResourceManager:
    """
    Centralized resource management for models, datasets, and temporary files
    that works across different operating systems.

    This class provides thread-safe operations for managing application resources
    including models, datasets, cache files, and temporary workspaces.

    Attributes:
        app_name (str): Application name used for directory naming.
        user_data_dir (Path): User-specific data directory for persistent storage.
        user_cache_dir (Path): User-specific cache directory for temporary files.
        temp_dir (Path): Temporary directory for runtime operations.
        model_dir (Path): Directory for storing model files.
        dataset_dir (Path): Directory for storing dataset files.
        downloads_dir (Path): Directory for downloaded files.
    """

    def __init__(self, app_name: str | None = None, custom_base_dir: str | Path | None = None):
        """
        Initialize resource paths with cross-platform compatibility.

        Args:
            app_name: Custom application name. If None, attempts to load from pyproject.toml.
            custom_base_dir: Custom base directory for all resources. If None, uses system defaults.

        Raises:
            ResourceManagerError: If initialization fails.
        """
        self._lock = Lock()
        self._workspace_registry: dict[str, Path] = {}

        # Determine application name
        self.app_name = self._determine_app_name(app_name)

        # Initialize directory paths
        self._initialize_paths(custom_base_dir)

        # Create necessary directories
        self._initialize_directories()

        logger.info(f"ResourceManager initialized for app: {self.app_name}")
        logger.debug(f"Resource directories: {self.get_all_directories()}")

    def _determine_app_name(self, app_name: str | None = None) -> str:
        """
        Determine the application name from various sources.

        Args:
            app_name: Explicitly provided app name.

        Returns:
            Determined application name.
        """
        if app_name:
            return app_name

        # Try to load from pyproject.toml
        try:
            project_root = self._find_project_root()
            config_path = project_root / "pyproject.toml"

            if config_path.exists():
                config = toml.load(str(config_path))
                name = config.get("project", {}).get("name")
                if name:
                    return name

                # Fallback to tool.poetry.name for poetry projects
                name = config.get("tool", {}).get("poetry", {}).get("name")
                if name:
                    return name

        except Exception as e:
            logger.warning(f"Could not load app name from pyproject.toml: {e}")

        # Fallback to default
        return "culicidaelab"

    def _find_project_root(self) -> Path:
        """
        Find the project root directory by looking for configuration files.

        Returns:
            Path to the project root directory.

        Raises:
            ResourceManagerError: If project root cannot be found.
        """
        current_path = Path(__file__).resolve()

        # Look for common project indicators
        indicators = ["pyproject.toml", "setup.py", "setup.cfg", ".git", "requirements.txt"]

        while current_path.parent != current_path:  # Stop at filesystem root
            if any((current_path / indicator).exists() for indicator in indicators):
                return current_path
            current_path = current_path.parent

        # If we can't find project root, use the directory containing this file
        logger.warning("Could not find project root, using module directory")
        return Path(__file__).parent.parent

    def _initialize_paths(self, custom_base_dir: str | Path | None = None) -> None:
        """
        Initialize all resource paths.

        Args:
            custom_base_dir: Custom base directory for all resources.
        """
        if custom_base_dir:
            base_dir = Path(custom_base_dir).resolve()
            self.user_data_dir = base_dir / "data"
            self.user_cache_dir = base_dir / "cache"
        else:
            # Use platform-appropriate directories
            self.user_data_dir = Path(appdirs.user_data_dir(self.app_name))
            self.user_cache_dir = Path(appdirs.user_cache_dir(self.app_name))

        # Temporary directory for runtime operations
        self.temp_dir = Path(tempfile.gettempdir()) / self.app_name

        # Subdirectories for different resource types
        self.model_dir = self.user_data_dir / "models"
        self.dataset_dir = self.user_data_dir / "datasets"
        self.downloads_dir = self.user_data_dir / "downloads"
        self.logs_dir = self.user_data_dir / "logs"
        self.config_dir = self.user_data_dir / "config"

    def _initialize_directories(self) -> None:
        """
        Create necessary directories with proper permissions.

        Raises:
            ResourceManagerError: If directory creation fails.
        """
        directories = [
            self.user_data_dir,
            self.user_cache_dir,
            self.model_dir,
            self.dataset_dir,
            self.downloads_dir,
            self.logs_dir,
            self.config_dir,
            self.temp_dir,
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created/verified directory: {directory}")
            except Exception as e:
                raise ResourceManagerError(f"Failed to create directory {directory}: {e}") from e

        # Set appropriate permissions (Unix-like systems only)
        if platform.system() != "Windows":
            self._set_directory_permissions(directories)

    def _set_directory_permissions(self, directories: list[Path]) -> None:
        """
        Set appropriate permissions for directories on Unix-like systems.

        Args:
            directories: List of directories to set permissions for.
        """
        try:
            for directory in directories:
                os.chmod(directory, 0o755)
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")

    def get_model_path(self, model_name: str, create_if_missing: bool = True) -> Path:
        """
        Get a standardized path for a specific model.

        Args:
            model_name: Name of the model.
            create_if_missing: Whether to create the directory if it doesn't exist.

        Returns:
            Absolute path to the model directory.

        Raises:
            ResourceManagerError: If path creation fails.
        """
        if not model_name or not model_name.strip():
            raise ValueError("Model name cannot be empty")

        model_path = self.model_dir / self._sanitize_name(model_name)

        if create_if_missing:
            try:
                model_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ResourceManagerError(f"Failed to create model directory {model_path}: {e}") from e

        return model_path

    def get_dataset_path(self, dataset_name: str, create_if_missing: bool = True) -> Path:
        """
        Get a standardized path for a specific dataset.

        Args:
            dataset_name: Name of the dataset.
            create_if_missing: Whether to create the directory if it doesn't exist.

        Returns:
            Absolute path to the dataset directory.

        Raises:
            ResourceManagerError: If path creation fails.
        """
        if not dataset_name or not dataset_name.strip():
            raise ValueError("Dataset name cannot be empty")

        dataset_path = self.dataset_dir / self._sanitize_name(dataset_name)

        if create_if_missing:
            try:
                dataset_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ResourceManagerError(f"Failed to create dataset directory {dataset_path}: {e}") from e

        return dataset_path

    def get_cache_path(self, cache_name: str, create_if_missing: bool = True) -> Path:
        """
        Get a path for cache files.

        Args:
            cache_name: Name of the cache.
            create_if_missing: Whether to create the directory if it doesn't exist.

        Returns:
            Path to the cache directory.

        Raises:
            ResourceManagerError: If path creation fails.
        """
        if not cache_name or not cache_name.strip():
            raise ValueError("Cache name cannot be empty")

        cache_path = self.user_cache_dir / self._sanitize_name(cache_name)

        if create_if_missing:
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ResourceManagerError(f"Failed to create cache directory {cache_path}: {e}") from e

        return cache_path

    def create_temp_workspace(self, prefix: str = "workspace", suffix: str = "") -> Path:
        """
        Create a temporary workspace for runtime operations.

        Args:
            prefix: Prefix for the temporary directory name.
            suffix: Suffix for the temporary directory name.

        Returns:
            Path to the created temporary workspace.

        Raises:
            ResourceManagerError: If workspace creation fails.
        """
        try:
            # Create unique workspace name
            timestamp = str(int(time.time()))
            pid = str(os.getpid())
            workspace_name = f"{prefix}_{timestamp}_{pid}"

            if suffix:
                workspace_name += f"_{suffix}"

            temp_workspace = self.temp_dir / workspace_name
            temp_workspace.mkdir(parents=True, exist_ok=True)

            # Register workspace for cleanup tracking
            with self._lock:
                self._workspace_registry[workspace_name] = temp_workspace

            logger.info(f"Created temporary workspace: {temp_workspace}")
            return temp_workspace

        except Exception as e:
            raise ResourceManagerError(f"Failed to create temporary workspace: {e}") from e

    @contextmanager
    def temp_workspace(self, prefix: str = "workspace", suffix: str = ""):
        """
        Context manager for temporary workspaces that auto-cleans on exit.

        Args:
            prefix: Prefix for the temporary directory name.
            suffix: Suffix for the temporary directory name.

        Yields:
            Path to the temporary workspace.

        Example:
            with resource_manager.temp_workspace("processing") as workspace:
                # Use workspace for temporary operations
                temp_file = workspace / "temp.txt"
                temp_file.write_text("temporary data")
            # Workspace is automatically cleaned up here
        """
        workspace = self.create_temp_workspace(prefix, suffix)
        try:
            yield workspace
        finally:
            self.clean_temp_workspace(workspace, force=True)

    def clean_temp_workspace(self, workspace_path: Path, force: bool = False) -> None:
        """
        Clean up a temporary workspace.

        Args:
            workspace_path: Path to the workspace to clean.
            force: If True, force remove even if not in temp directory.

        Raises:
            ResourceManagerError: If cleanup fails or workspace is outside temp directory.
        """
        try:
            # Safety check to prevent accidental deletion
            if not force and not self._is_safe_to_delete(workspace_path):
                raise ValueError("Cannot clean workspace outside of temp directory without force=True")

            if workspace_path.exists():
                if workspace_path.is_dir():
                    shutil.rmtree(workspace_path)
                else:
                    workspace_path.unlink()

                logger.info(f"Cleaned workspace: {workspace_path}")

            # Remove from registry
            with self._lock:
                workspace_name = workspace_path.name
                self._workspace_registry.pop(workspace_name, None)

        except Exception as e:
            raise ResourceManagerError(f"Failed to clean workspace {workspace_path}: {e}") from e

    def _is_safe_to_delete(self, path: Path) -> bool:
        """
        Check if a path is safe to delete (within managed directories).

        Args:
            path: Path to check.

        Returns:
            True if path is safe to delete.
        """
        safe_parents = [self.temp_dir, self.user_cache_dir]

        try:
            resolved_path = path.resolve()
            return any(str(resolved_path).startswith(str(parent.resolve())) for parent in safe_parents)
        except Exception:
            return False

    def clean_old_files(self, days: int = 5, include_cache: bool = True) -> dict[str, int]:
        """
        Clean up old download and temporary files.

        Args:
            days: Number of days after which files are considered old.
            include_cache: Whether to include cache directory in cleanup.

        Returns:
            Dictionary with cleanup statistics.
        """
        if days < 0:
            raise ValueError("Days must be non-negative")

        cleanup_stats = {
            "downloads_cleaned": 0,
            "temp_cleaned": 0,
            "cache_cleaned": 0,
            "errors": 0,
        }

        cutoff_time = time.time() - (days * 86400)

        # Clean downloads directory
        cleanup_stats["downloads_cleaned"] = self._clean_directory(
            self.downloads_dir,
            cutoff_time,
        )

        # Clean temporary directory
        cleanup_stats["temp_cleaned"] = self._clean_directory(
            self.temp_dir,
            cutoff_time,
        )

        # Clean cache directory if requested
        if include_cache:
            cleanup_stats["cache_cleaned"] = self._clean_directory(
                self.user_cache_dir,
                cutoff_time,
            )

        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats

    def _clean_directory(self, directory: Path, cutoff_time: float) -> int:
        """
        Clean files older than cutoff time in a directory.

        Args:
            directory: Directory to clean.
            cutoff_time: Cutoff time as timestamp.

        Returns:
            Number of items cleaned.
        """
        cleaned_count = 0

        if not directory.exists():
            return cleaned_count

        try:
            for item in directory.iterdir():
                try:
                    if item.stat().st_mtime < cutoff_time:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        cleaned_count += 1
                        logger.debug(f"Cleaned old file/directory: {item}")
                except Exception as e:
                    logger.warning(f"Could not remove {item}: {e}")

        except Exception as e:
            logger.error(f"Error cleaning directory {directory}: {e}")

        return cleaned_count

    def get_disk_usage(self) -> dict[str, dict[str, int | str]]:
        """
        Get disk usage statistics for all managed directories.

        Returns:
            Dictionary with disk usage information for each directory.
        """
        usage_stats = {}

        directories = {
            "user_data": self.user_data_dir,
            "cache": self.user_cache_dir,
            "models": self.model_dir,
            "datasets": self.dataset_dir,
            "downloads": self.downloads_dir,
            "temp": self.temp_dir,
        }

        for name, path in directories.items():
            usage_stats[name] = self._get_directory_size(path)

        return usage_stats

    def _get_directory_size(self, path: Path) -> dict[str, int | str]:
        """
        Get size information for a directory.

        Args:
            path: Directory path.

        Returns:
            Dictionary with size information.
        """
        if not path.exists():
            return {"size_bytes": 0, "size_human": "0 B", "file_count": 0}

        total_size = 0
        file_count = 0

        try:
            for item in path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
        except Exception as e:
            logger.warning(f"Error calculating size for {path}: {e}")

        return {
            "size_bytes": total_size,
            "size_human": self._format_bytes(total_size),
            "file_count": file_count,
        }

    def _format_bytes(self, bytes_count: int) -> str:
        """
        Format bytes into human-readable string.

        Args:
            bytes_count: Number of bytes.

        Returns:
            Human-readable size string.
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"

    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize a name for use as a directory/file name.

        Args:
            name: Original name.

        Returns:
            Sanitized name safe for filesystem use.
        """
        # Remove or replace unsafe characters
        import re

        sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
        sanitized = sanitized.strip(". ")  # Remove leading/trailing dots and spaces

        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed"

        return sanitized

    def create_checksum(self, file_path: str | Path, algorithm: str = "md5") -> str:
        """
        Create a checksum for a file.

        Args:
            file_path: Path to the file.
            algorithm: Hashing algorithm to use (md5, sha1, sha256).

        Returns:
            Hexadecimal checksum string.

        Raises:
            ResourceManagerError: If checksum creation fails.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise ResourceManagerError(f"File does not exist: {file_path}")

        try:
            hash_obj = hashlib.new(algorithm)

            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()

        except Exception as e:
            raise ResourceManagerError(f"Failed to create checksum for {file_path}: {e}") from e

    def verify_checksum(self, file_path: str | Path, expected_checksum: str, algorithm: str = "md5") -> bool:
        """
        Verify a file's checksum.

        Args:
            file_path: Path to the file.
            expected_checksum: Expected checksum value.
            algorithm: Hashing algorithm used.

        Returns:
            True if checksum matches, False otherwise.
        """
        try:
            actual_checksum = self.create_checksum(file_path, algorithm)
            return actual_checksum.lower() == expected_checksum.lower()
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

    def get_all_directories(self) -> dict[str, Path]:
        """
        Get all managed directories.

        Returns:
            Dictionary mapping directory names to paths.
        """
        return {
            "user_data_dir": self.user_data_dir,
            "user_cache_dir": self.user_cache_dir,
            "temp_dir": self.temp_dir,
            "model_dir": self.model_dir,
            "dataset_dir": self.dataset_dir,
            "downloads_dir": self.downloads_dir,
            "logs_dir": self.logs_dir,
            "config_dir": self.config_dir,
        }

    def __repr__(self) -> str:
        """String representation of ResourceManager."""
        return f"ResourceManager(app_name='{self.app_name}', user_data_dir='{self.user_data_dir}')"
