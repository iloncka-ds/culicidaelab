"""Remote URL Provider implementation."""

from typing import Any
from pathlib import Path

from ..core.base_provider import BaseProvider
from ..core.config_manager import ConfigManager
from ..core.utils import download_file


class RemoteURLProvider(BaseProvider):
    """Provider for downloading files from any public URL."""

    def __init__(self, config_manager: ConfigManager):
        """Initialize Remote URL provider.

        Args:
            config_manager (ConfigManager): Configuration manager instance
        """
        self.provider_name = "remote_url"
        self.config_manager = config_manager
        self.downloads_dir = self.config_manager.get_downloads_dir()

        # Optional: Get provider configuration if needed
        provider_config = self.config_manager.get_provider_config("remote_url", {})

    def get_metadata(self, url: str) -> dict[str, Any]:
        """Get metadata for a remote URL.

        Args:
            url (str): URL of the file to get metadata for

        Returns:
            Dict[str, Any]: File metadata
        """
        import requests

        try:
            response = requests.head(url)
            response.raise_for_status()

            return {
                "content_type": response.headers.get("Content-Type", "application/octet-stream"),
                "content_length": int(response.headers.get("Content-Length", 0)),
                "last_modified": response.headers.get("Last-Modified"),
                "url": url,
            }
        except requests.RequestException as e:
            # Log or handle metadata retrieval errors
            return {
                "error": str(e),
                "url": url,
            }

    def download_dataset(
        self,
        url: str,
        destination: Path | None = None,
        **kwargs,
    ) -> Path:
        """Download a file from a remote URL.

        Args:
            url (str): URL of the file to download
            destination (Optional[Path]): Optional destination path
            **kwargs: Additional download parameters

        Returns:
            Path: Path to the downloaded file
        """
        # Use download_file utility with optional downloads directory
        return download_file(
            url,
            destination=destination,
            downloads_dir=self.downloads_dir,
            **kwargs,
        )
