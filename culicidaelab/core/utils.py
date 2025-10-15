"""Utility functions for common operations.

This module contains helper functions used across the library, such as
downloading files and converting colors.
"""

import logging
from pathlib import Path
from collections.abc import Callable

import requests

import re
import tqdm
import uuid


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
    Downloads a file from the specified URL showing a progress bar and optionally calling
    a progress callback function. Supports both direct destination paths and default
    download directories.

    Args:
        url (str): The URL of the file to download. Must start with 'http://' or 'https://'.
        destination (Union[str, Path, None], optional): The complete file path where the
            downloaded file should be saved. If None, the file will be saved in downloads_dir
            with its original filename. Defaults to None.
        downloads_dir (Union[str, Path, None], optional): The directory to save the file in
            when no specific destination is provided. If None, uses current working directory.
            Defaults to None.
        progress_callback (Optional[Callable[[int, int], None]], optional): A function to call
            with progress updates. Takes two parameters: bytes downloaded and total bytes.
            Defaults to None.
        chunk_size (int, optional): Size of chunks to download in bytes. Larger chunks use
            more memory but may download faster. Defaults to 8192.
        timeout (int, optional): Number of seconds to wait for server response before timing
            out. Defaults to 30.
        desc (Optional[str], optional): Custom description for the progress bar. If None,
            uses the filename. Defaults to None.

    Returns:
        Path: Path object pointing to the downloaded file.

    Raises:
        ValueError: If the URL is invalid or doesn't start with http(s).
        RuntimeError: If the download fails due to network issues or if writing the file
            fails due to permission or disk space issues.

    Example:
        >>> from pathlib import Path
        >>> # Basic download to current directory
        >>> path = download_file('https://example.com/data.csv')
        >>> print(path)
        PosixPath('data.csv')

        >>> # Download with custom progress tracking
        >>> def progress(current, total):
        ...     print(f'Downloaded {current}/{total} bytes')
        >>> path = download_file(
        ...     'https://example.com/large_file.zip',
        ...     destination='downloads/myfile.zip',
        ...     progress_callback=progress
        ... )
    """
    if not url or not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url}")

    dest_path = Path(destination) if destination else None
    if dest_path is None:
        base_dir = Path(downloads_dir) if downloads_dir else Path.cwd()
        base_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        dest_path = base_dir / filename

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            progress_desc = desc or f"Downloading {dest_path.name}"

            with tqdm.tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=progress_desc,
            ) as pbar:
                with open(dest_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        written_size = file.write(chunk)
                        pbar.update(written_size)
                        if progress_callback:
                            try:
                                progress_callback(pbar.n, total_size)
                            except Exception as cb_err:
                                logging.warning(f"Progress callback error: {cb_err}")
        return dest_path
    except requests.RequestException as e:
        logging.error(f"Download failed for {url}: {e}")
        raise RuntimeError(f"Failed to download file from {url}: {e}") from e
    except OSError as e:
        logging.error(f"File write error for {dest_path}: {e}")
        raise RuntimeError(f"Failed to write file to {dest_path}: {e}") from e


def create_safe_path(name: str) -> str:
    """
    Sanitize a string to create a safe directory or file name.

    Converts a string into a valid filename by replacing Windows/Unix reserved characters
    with underscores and removing potentially problematic leading/trailing characters.
    If the resulting string is empty, returns a UUID.

    Args:
        name (str): The string to be converted into a safe filename. If not a string,
            it will be converted to one.

    Returns:
        str: A sanitized string safe for use as a filename or directory name.
            - Reserved characters (<>:\"/\\|?*) are replaced with underscores
            - Leading/trailing dots and whitespace are removed
            - If result would be empty, returns a UUID

    Example:
        >>> # Basic filename sanitization
        >>> create_safe_path('my:file*.txt')
        'my_file_.txt'

        >>> # Handling special characters and spaces
        >>> create_safe_path('data/file (1).csv')
        'data_file (1).csv'

        >>> # Empty or invalid input
        >>> path = create_safe_path('')  # Returns a UUID
        >>> print(len(path))
        36
    """

    if not isinstance(name, str):
        name = str(name)

    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name).strip(". ")
    return sanitized or str(uuid.uuid4())
