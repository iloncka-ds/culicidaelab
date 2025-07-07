import os
import time
import json
import csv
import psutil
import tracemalloc
import GPUtil
import numpy as np
from pathlib import Path
from typing import Any

from collections.abc import Callable

from culicidaelab.core.base_predictor import BasePredictor
from culicidaelab.core.settings import Settings
from culicidaelab.predictors.classifier import MosquitoClassifier
from culicidaelab.predictors.detector import MosquitoDetector


def measure_performance(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_start = {gpu.id: gpu.memoryUsed for gpu in gpus}
    else:
        gpu_start = None

    process = psutil.Process(os.getpid())
    cpu_start = process.cpu_percent(interval=None)
    mem_start = process.memory_info().rss
    tracemalloc.start()
    start_time = time.time()

    result = func(*args, **kwargs)

    end_time = time.time()
    cpu_end = process.cpu_percent(interval=None)
    mem_end = process.memory_info().rss
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gpu_end = {gpu.id: gpu.memoryUsed for gpu in GPUtil.getGPUs()} if gpus else None
    if gpu_start and gpu_end:
        gpu_memory_diff_mb = {gpu_id: gpu_end[gpu_id] - gpu_start.get(gpu_id, 0) for gpu_id in gpu_end}
    else:
        gpu_memory_diff_mb = None
    return {
        "result": result,
        "time_seconds": end_time - start_time,
        "cpu_percent": cpu_end - cpu_start,
        "memory_rss_bytes": mem_end - mem_start,
        "tracemalloc_peak_bytes": peak,
        "gpu_memory_diff_mb": gpu_memory_diff_mb,
    }


def save_results(results: dict[str, Any], path: Path, fmt: str = "json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
    elif fmt == "csv":
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for key, value in results.items():
                writer.writerow([key, value])


def run_batch_test(
    model_name: str,
    model: BasePredictor,
    inputs: list[np.ndarray],
    ground_truths: list[Any],
    output_path: Path,
):
    results = measure_performance(model.evaluate_batch, inputs, ground_truths)
    metrics = results.pop("result")
    results.update(metrics)
    save_results(results, output_path.with_suffix(".json"), fmt="json")
    save_results(results, output_path.with_suffix(".csv"), fmt="csv")
    print(f"{model_name.capitalize()} performance logged at {output_path}")


if __name__ == "__main__":
    settings = Settings()

    # Create a mock weights manager for testing
    from culicidaelab.core.provider_service import ProviderService
    from culicidaelab.predictors.model_weights_manager import ModelWeightsManager

    provider_service = ProviderService(settings=settings)
    weights_manager = ModelWeightsManager(
        settings=settings,
        provider_service=provider_service,
    )

    classifier = MosquitoClassifier(settings, load_model=True)
    detector = MosquitoDetector(settings, load_model=True)

    inputs = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(8)]
    cls_truths = ["species_a"] * len(inputs)
    det_truths = [[(256.0, 256.0, 100.0, 100.0)]] * len(inputs)

    run_batch_test(
        "classification",
        classifier,
        inputs,
        cls_truths,
        Path("logs/classification"),
    )
    run_batch_test("detection", detector, inputs, det_truths, Path("logs/detection"))
