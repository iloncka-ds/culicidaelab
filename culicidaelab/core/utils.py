"""Utility functions for common operations.

This module contains helper functions used across the library, such as
downloading files and converting colors.
"""

import logging
from pathlib import Path
from collections.abc import Callable

import requests
import tqdm
from culicidaelab.core.config_models import PredictorConfig


def download_file(
    url: str,
    destination: str | Path | None = None,
    downloads_dir: str | Path | None = None,
    progress_callback: Callable | None = None,
    chunk_size: int = 8192,
    timeout: int = 30,
    desc: str | None = None,
) -> Path:
    """Downloads a file from a URL with progress tracking.

    Args:
        url (str): The URL of the file to download.
        destination (str | Path, optional): The specific destination path for the file.
        downloads_dir (str | Path, optional): Default directory for downloads.
        progress_callback (Callable, optional): A custom progress callback.
        chunk_size (int): The size of chunks to download in bytes.
        timeout (int): The timeout for the download request in seconds.
        desc (str, optional): A description for the progress bar.

    Returns:
        Path: The path to the downloaded file.

    Raises:
        ValueError: If the URL is invalid.
        RuntimeError: If the download or file write fails.
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


def construct_weights_path(
    model_dir: Path,
    predictor_config: PredictorConfig,
    backend: str | None = None,
) -> Path:
    """
    A pure, static function to construct a model weights path from config data.
    This function has no dependencies on Settings or Manager instances.
    Args:
        model_dir (Path): The Path of the model directory (local folder).
        predictor_config (PredictorConfig): Predictor config
        backend (str | None, optional): Backend


    Returns:
        Path: The path to the model weights file.

    Raises:
        ValueError: If no backend specified for model.

    """
    final_backend = backend if backend is not None else predictor_config.backend
    if not final_backend:
        raise ValueError("No backend specified for model.")

    if not predictor_config.weights or final_backend not in predictor_config.weights:
        raise ValueError(f"Backend '{final_backend}' not defined in weights config.")

    filename = predictor_config.weights[final_backend].filename
    return model_dir / filename
