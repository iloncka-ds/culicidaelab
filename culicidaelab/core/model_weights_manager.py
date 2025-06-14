# """
# Model weights management module for CulicidaeLab.
# """

# from __future__ import annotations
# import shutil
# from pathlib import Path

# from huggingface_hub import hf_hub_download

# from culicidaelab.core.settings import Settings


# class ModelWeightsManager:
#     """Manages ensuring model weights are available locally, downloading if necessary."""

#     def __init__(self, settings: Settings):
#         """Initialize the model weights manager."""
#         self.settings = settings

#     def ensure_weights(self, model_type: str) -> Path:
#         """
#         Ensures weights for a model type exist locally, downloading if not.
#         This method correctly handles and resolves symbolic links.

#         Args:
#             model_type: The key for the predictor (e.g., 'classifier').

#         Returns:
#             The absolute, canonical Path to the validated, existing model file.
#         """
#         local_path = self.settings.get_model_weights(model_type).resolve()

#         if local_path.exists():
#             if local_path.is_symlink():
#                 try:
#                     real_path = local_path.resolve(strict=True)
#                     print(f"Symlink found at {local_path}, resolved to real file: {real_path}")
#                     return real_path
#                 except FileNotFoundError:
#                     print(f"Warning: Broken symlink found at {local_path}. It will be removed.")
#                     local_path.unlink()
#             else:
#                 print(f"Weights file found at: {local_path}")
#                 return local_path

#         # If we get here, the path either does not exist or was a broken symlink.
#         # We must now download the file.
#         print(f"Model weights for '{model_type}' not found. Attempting to download...")

#         predictor_config = self.settings.get_config(f'predictors.{model_type}')
#         repo_id = predictor_config.repository_id
#         filename = predictor_config.filename

#         if not repo_id or not filename:
#             raise ValueError(
#                 f"Cannot download weights for '{model_type}'. "
#                 f"Configuration is missing 'repository_id' or 'filename'. "
#                 f"Please place the file manually at: {local_path}"
#             )

#         try:
#             dest_dir = local_path.parent.resolve()
#             print(f"Ensuring destination directory exists: {dest_dir}")

#             # Robust directory creation with validation
#             dest_dir.mkdir(parents=True, exist_ok=True)
#             if not dest_dir.is_dir():
#                 raise NotADirectoryError(f"Failed to create directory: {dest_dir}")

#             # Download the file to the Hugging Face cache
#             downloaded_path_str = hf_hub_download(
#                 repo_id=repo_id,
#                 filename=filename,
#                 cache_dir=self.settings.cache_dir / "huggingface",
#                 local_dir=str(local_path.parent),
#                 local_dir_use_symlinks=False,

#             )

#             # Now we can safely copy the file.
#             print(f"Copying weights to: {local_path}")
#             shutil.copy(downloaded_path_str, str(local_path))

#             print(f"Successfully downloaded weights to: {local_path}")
#             return local_path
#         except Exception as e:
#             # Clean up a potentially partial file on failure
#             if local_path.exists():
#                 local_path.unlink()
#             # Include directory info in error message
#             dir_status = "exists" if dest_dir.exists() else "missing"
#             dir_type = "directory" if dest_dir.is_dir() else "not-a-directory"
#             raise RuntimeError(
#                 f"Failed to download weights for '{model_type}' to {local_path}. "
#                 f"Directory status: {dir_status} ({dir_type}). Error: {e}"
#             ) from e

"""
Model weights management module for CulicidaeLab.
"""

from __future__ import annotations
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from culicidaelab.core.settings import Settings


class ModelWeightsManager:
    """Manages ensuring model weights are available locally, downloading if necessary."""

    def __init__(self, settings: Settings):
        """Initialize the model weights manager."""
        self.settings = settings

    def ensure_weights(self, model_type: str) -> Path:
        """
        Ensures weights for a model type exist locally, downloading if not.
        This method correctly handles and resolves symbolic links.

        Args:
            model_type: The key for the predictor (e.g., 'classifier').

        Returns:
            The absolute, canonical Path to the validated, existing model file.
        """
        local_path = self.settings.get_model_weights(model_type).resolve()

        if local_path.exists():
            if local_path.is_symlink():
                try:
                    real_path = local_path.resolve(strict=True)
                    print(f"Symlink found at {local_path}, resolved to real file: {real_path}")
                    return real_path
                except FileNotFoundError:
                    print(f"Warning: Broken symlink found at {local_path}. It will be removed.")
                    local_path.unlink()
            else:
                print(f"Weights file found at: {local_path}")
                return local_path

        # If we get here, the path either does not exist or was a broken symlink.
        # We must now download the file.
        print(f"Model weights for '{model_type}' not found. Attempting to download...")

        predictor_config = self.settings.get_config(f'predictors.{model_type}')
        repo_id = predictor_config.repository_id
        filename = predictor_config.filename

        if not repo_id or not filename:
            raise ValueError(
                f"Cannot download weights for '{model_type}'. "
                f"Configuration is missing 'repository_id' or 'filename'. "
                f"Please place the file manually at: {local_path}"
            )

        try:
            dest_dir = local_path.parent.resolve()
            print(f"Ensuring destination directory exists: {dest_dir}")

            # Robust directory creation with validation
            dest_dir.mkdir(parents=True, exist_ok=True)
            if not dest_dir.is_dir():
                raise NotADirectoryError(f"Failed to create directory: {dest_dir}")

            # Download the file to the Hugging Face cache
            downloaded_path_str = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.settings.cache_dir / "huggingface",
                local_dir=str(local_path.parent),
                )
            print(f"Downloaded weights to: {downloaded_path_str}")
            if model_type == "segmenter":
                # For segmenter, we need to ensure the file is not a symlink
                downloaded_yaml = hf_hub_download(
                repo_id=repo_id,
                filename=predictor_config.sam_config_filename,
                cache_dir=self.settings.cache_dir / "huggingface",
                local_dir=str(local_path.parent),
                )
                print(f"Downloaded SAM config to: {downloaded_yaml}")
            # # Now we can safely copy the file.
            # print(f"Copying weights to: {local_path}")
            # shutil.copy(downloaded_path_str, str(local_path))

            print(f"Successfully downloaded weights to: {local_path}")
            return local_path
        except Exception as e:
            # Clean up a potentially partial file on failure
            if local_path.exists():
                local_path.unlink()
            # Include directory info in error message
            dir_status = "exists" if dest_dir.exists() else "missing"
            dir_type = "directory" if dest_dir.is_dir() else "not-a-directory"
            raise RuntimeError(
                f"Failed to download weights for '{model_type}' to {local_path}. "
                f"Directory status: {dir_status} ({dir_type}). Error: {e}"
            ) from e
