from __future__ import annotations
from typing import TYPE_CHECKING
import importlib.util

if TYPE_CHECKING:
    from culicidaelab.core.settings import Settings
    from culicidaelab.core.base_inference_backend import BaseInferenceBackend


def create_backend(
    settings: Settings,
    predictor_type: str,
    mode: str | None = None,
) -> BaseInferenceBackend:
    """
    Intelligently creates and returns an inference backend instance.

    This factory is the core of the library's adaptable architecture. It ensures
    that the correct backend is chosen based on a clear order of precedence,
    and it gracefully handles different installation environments (e.g., a minimal
    '[serve]' installation vs. a full '[torch]' installation).

    The selection logic follows this strict order of precedence:
    1.  **Code Override (`mode` parameter):** The `mode` parameter, if provided,
        is the highest priority and will force a backend choice.
        - `mode='serve'` forces the 'onnx' backend.
        - `mode='experiments'` (or any other string) forces the 'torch' backend.

    2.  **Configuration Override (YAML file):** If `mode` is not set, the factory
        checks for a `backend` key in the predictor's YAML configuration file
        (e.g., `conf/predictors/classifier.yaml`).

    3.  **Environment Auto-Detection:** If neither of the above is specified,
        the factory inspects the environment. It checks if `torch` is installed.
        - If `torch` is present, it defaults to the 'torch' backend.
        - If `torch` is NOT present (a '[serve]' installation), it safely
          defaults to the 'onnx' backend.

    Args:
        predictor_type: The type of predictor requesting a backend
                        (e.g., 'classifier', 'detector').
        settings: The main library settings object.
        mode: An optional, high-priority override to force a specific backend.

    Returns:
        An instantiated backend object that conforms to the InferenceBackendProtocol.

    Raises:
        RuntimeError: If the user explicitly requests a 'torch' backend but does
                      not have the required libraries installed.
        ValueError: If a backend cannot be resolved for the given predictor type.
    """
    user_explicit_choice = None

    # --- Step 1: Determine user's explicit intent (Code > Config) ---
    if mode:
        user_explicit_choice = "onnx" if mode == "serve" else "torch"
    else:
        config_backend = settings.get_config(f"predictors.{predictor_type}.backend")
        if config_backend in ["onnx", "torch"]:
            user_explicit_choice = config_backend

    # --- Step 2: Act as a Gatekeeper ---
    # If the user explicitly requested 'torch', we must validate their environment.
    # This prevents ImportError crashes in a '[serve]' installation.
    if user_explicit_choice == "torch":
        if not importlib.util.find_spec("torch"):
            raise RuntimeError(
                "The 'torch' backend was requested, but PyTorch is not installed. "
                "Please install the full library with: pip install 'culicidaelab[full]'",
            )

    # --- Step 3: Determine the final backend to load ---
    final_backend_type = user_explicit_choice
    # If no explicit choice was made, perform environment auto-detection.
    if not final_backend_type:
        if importlib.util.find_spec("torch"):
            final_backend_type = "torch"  # Default to torch if available
        else:
            final_backend_type = "onnx"  # Safely default to onnx if not

    # --- Step 4: Instantiate and return the chosen backend ---
    # The ModelWeightsManager is now created here and injected into the backend.
    from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

    weights_manager = ModelWeightsManager(settings)
    predictor_config = settings.get_config(f"predictors.{predictor_type}")

    if final_backend_type == "torch":
        if predictor_type == "classifier":
            from culicidaelab.predictors.backends.classifier._fastai import ClassifierFastAIBackend

            return ClassifierFastAIBackend(weights_manager=weights_manager)
        elif predictor_type == "detector":
            from culicidaelab.predictors.backends.detector._yolo import DetectorYOLOBackend

            return DetectorYOLOBackend(weights_manager=weights_manager)
        elif predictor_type == "segmenter":
            from culicidaelab.predictors.backends.segmenter._sam import SegmenterSAMBackend

            return SegmenterSAMBackend(weights_manager=weights_manager)

    elif final_backend_type == "onnx":
        if predictor_type == "classifier":
            from culicidaelab.predictors.backends.classifier._onnx import ClassifierONNXBackend

            return ClassifierONNXBackend(weights_manager=weights_manager, config=predictor_config)
        elif predictor_type == "detector":
            from culicidaelab.predictors.backends.detector._onnx import DetectorONNXBackend

            return DetectorONNXBackend(weights_manager=weights_manager)
        elif predictor_type == "segmenter":
            from culicidaelab.predictors.backends.segmenter._onnx import SegmenterONNXBackend

            return SegmenterONNXBackend(weights_manager=weights_manager)

    # This is a safeguard for future development.
    raise ValueError(f"Could not create a backend for predictor '{predictor_type}' with mode '{final_backend_type}'.")
