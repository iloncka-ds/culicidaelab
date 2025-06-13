"""Utility functions for common operations."""

from pathlib import Path
import requests
from collections.abc import Callable
import logging
import tqdm


def download_file(
    url: str,
    destination: str | Path | None = None,
    downloads_dir: str | Path | None = None,
    progress_callback: Callable | None = None,
    chunk_size: int = 8192,
    timeout: int = 30,
    desc: str | None = None,
) -> Path:
    """
    Download a file from a given URL with optional progress tracking using tqdm.

    Args:
        url (str): URL of the file to download
        destination (Optional[Union[str, Path]]): Specific destination path for the file
        downloads_dir (Optional[Union[str, Path]]): Default directory for downloads if no destination specified
        progress_callback (Optional[Callable]): Optional custom progress callback
        chunk_size (int): Size of chunks to download (default: 8192 bytes)
        timeout (int): Timeout for the download request in seconds (default: 30)
        desc (Optional[str]): Description for the progress bar

    Returns:
        Path: Path to the downloaded file

    Raises:
        ValueError: If URL is invalid
        RuntimeError: If download fails
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url}")

    destination = Path(destination) if destination else None
    downloads_dir = Path(downloads_dir) if downloads_dir else None

    if destination is None:
        base_dir = downloads_dir or Path.cwd()

        base_dir.mkdir(parents=True, exist_ok=True)

        filename = url.split("/")[-1]
        destination = base_dir / filename

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            progress_desc = desc or f"Downloading {destination.name}"

            with tqdm.tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=progress_desc,
                disable=total_size == 0,
            ) as progress_bar:
                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        chunk_size = file.write(chunk)

                        progress_bar.update(chunk_size)

                        if progress_callback and callable(progress_callback):
                            try:
                                progress_callback(chunk_size, total_size)
                            except Exception as cb_error:
                                logging.warning(f"Progress callback error: {cb_error}")

        return destination

    except requests.RequestException as e:
        logging.error(f"Download failed for {url}: {e}")
        raise RuntimeError(f"Failed to download file from {url}: {e}") from e
    except OSError as e:
        logging.error(f"File write error for {destination}: {e}")
        raise RuntimeError(f"Failed to write file to {destination}: {e}") from e


def default_progress_callback(downloaded: int, total: int) -> None:
    """
    Default progress callback that can be used with download_file.

    Args:
        downloaded (int): Number of bytes downloaded
        total (int): Total number of bytes to download
    """
    if total > 0:
        percentage = (downloaded / total) * 100
        print(f"Download progress: {percentage:.2f}% ({downloaded}/{total} bytes)")
